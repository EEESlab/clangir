//===--- CIRGenStmtOpenMP.cpp - Emit MLIR Code from OpenMP Statements -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OpenMP Stmt nodes as MLIR code.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTFwd.h"
#include "clang/AST/StmtIterator.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/ParentMapContext.h" //lucap: added for debug reason, to print ast parent nodes
#include "clang/Basic/OpenMPKinds.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/raw_ostream.h"

#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace mlir::omp;

static void buildDependences(const OMPExecutableDirective &S,
                             OMPTaskDataTy &Data) {

  // First look for 'omp_all_memory' and add this first.
  bool OmpAllMemory = false;
  if (llvm::any_of(
          S.getClausesOfKind<OMPDependClause>(), [](const OMPDependClause *C) {
            return C->getDependencyKind() == OMPC_DEPEND_outallmemory ||
                   C->getDependencyKind() == OMPC_DEPEND_inoutallmemory;
          })) {
    OmpAllMemory = true;
    // Since both OMPC_DEPEND_outallmemory and OMPC_DEPEND_inoutallmemory are
    // equivalent to the runtime, always use OMPC_DEPEND_outallmemory to
    // simplify.
    OMPTaskDataTy::DependData &DD =
        Data.Dependences.emplace_back(OMPC_DEPEND_outallmemory,
                                      /*IteratorExpr=*/nullptr);
    // Add a nullptr Expr to simplify the codegen in emitDependData.
    DD.DepExprs.push_back(nullptr);
  }
  // Add remaining dependences skipping any 'out' or 'inout' if they are
  // overridden by 'omp_all_memory'.
  for (const auto *C : S.getClausesOfKind<OMPDependClause>()) {
    OpenMPDependClauseKind Kind = C->getDependencyKind();
    if (Kind == OMPC_DEPEND_outallmemory || Kind == OMPC_DEPEND_inoutallmemory)
      continue;
    if (OmpAllMemory && (Kind == OMPC_DEPEND_out || Kind == OMPC_DEPEND_inout))
      continue;
    OMPTaskDataTy::DependData &DD =
        Data.Dependences.emplace_back(C->getDependencyKind(), C->getModifier());
    DD.DepExprs.append(C->varlist_begin(), C->varlist_end());
  }
}

//===----------------------------------------------------------------------===//
// emit Parallel Directive
//===----------------------------------------------------------------------===//

mlir::LogicalResult
CIRGenFunction::emitOMPParallelDirective(const OMPParallelDirective &S) {

  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  // Create a `omp.parallel` op.
  auto parallelOp = ParallelOp::create(builder, scopeLoc);
  mlir::Block &block = parallelOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);
  // Create a scope for the OpenMP region.
  cir::ScopeOp::create(
      builder, scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (emitStmt(S.getCapturedStmt(OpenMPDirectiveKind::OMPD_parallel)
                         ->getCapturedStmt(),
                     /*useCurrentScope=*/true)
                .failed())
          res = mlir::failure();
      });
  // Add the terminator for `omp.parallel`.
  TerminatorOp::create(builder, getLoc(S.getSourceRange().getEnd()));
  return res;
}

//===----------------------------------------------------------------------===//
// emit Task Directives
//===----------------------------------------------------------------------===//

mlir::LogicalResult
CIRGenFunction::emitOMPTaskwaitDirective(const OMPTaskwaitDirective &S) {
  mlir::LogicalResult res = mlir::success();
  OMPTaskDataTy Data;
  buildDependences(S, Data);
  Data.HasNowaitClause = S.hasClausesOfKind<OMPNowaitClause>();
  CGM.getOpenMPRuntime().emitTaskWaitCall(builder, *this,
                                          getLoc(S.getSourceRange()), Data);
  return res;
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskyieldDirective(const OMPTaskyieldDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.taskyield operation
  CGM.getOpenMPRuntime().emitTaskyieldCall(builder, *this,
                                           getLoc(S.getSourceRange()));
  return res;
}

//===----------------------------------------------------------------------===//
// emit Barrier Directive
//===----------------------------------------------------------------------===//

mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.barrier operation
  CGM.getOpenMPRuntime().emitBarrierCall(builder, *this,
                                         getLoc(S.getSourceRange()));
  return res;
}


//===----------------------------------------------------------------------===//
// emit OMP For Directive
//===----------------------------------------------------------------------===//

// lucap: trying to implement OMP For Directive
// only emits the wsloop operation, the loop_nest will be emitted by visiting the ForStmt
mlir::LogicalResult   // return value is like a boolean, but more explicit (success / failure)
CIRGenFunction::emitOMPForDirective(const OMPForDirective &S) {   // pointer to Clang AST node

  // default set return value as success
  mlir::LogicalResult res = mlir::success();
  // retrieve metadata location of the Clang AST node
  auto scopeLoc = getLoc(S.getSourceRange());
  llvm::errs() << "DEBUG: creating omp.wsloop op\n";

  // Get the ForStmt - use getInnermostCapturedStmt() instead
  const CapturedStmt *CS = S.getInnermostCapturedStmt();
  const ForStmt *FS = dyn_cast<ForStmt>(CS->getCapturedStmt());

  // --- NEW: HOIST CONSTANTS HERE ---
  // The builder is currently OUTSIDE the wsloop. 
  // We create the constants NOW so they dominate the loop.
  mlir::Value lowerBound, upperBound, step;
  bool inclusive = false;
  
  if (FS) {
    // Handle Lower Bound
    if (const auto *DS = dyn_cast<DeclStmt>(FS->getInit())) {
        if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            if (const auto *IL = dyn_cast<IntegerLiteral>(VD->getInit()->IgnoreImpCasts())) {
                lowerBound = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, IL->getValue().getSExtValue());
            }
        }
    }


    // Handle Upper Bound and inclusive
    if (FS->getCond()) {
        if (const auto *BO = dyn_cast<BinaryOperator>(FS->getCond())) {
          // Get the RHS of the comparison as upper bound
          // upperBound = emitScalarExpr(BO->getRHS()); not working since emit a cir.int that is not accepted by omp.loop_nest
          // While it works evaluating RHS as a constant (same thing for LB and step)
          if (const auto *IL = dyn_cast<IntegerLiteral>(BO->getRHS()->IgnoreImpCasts())) {
            int64_t val = IL->getValue().getSExtValue();
            upperBound = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, val);
          } 

          // Check if comparison is inclusive (<= or >=) or exclusive (< or >)
          BinaryOperatorKind opKind = BO->getOpcode();
          if (opKind == BO_LE || opKind == BO_GE) {
            inclusive = true;
          }
        }
      }


    // 3. Get step from increment (e.g., i++, i+=2)
    if (FS->getInc()) {
      if (const auto *UO = dyn_cast<UnaryOperator>(FS->getInc())) {
        // i++ or ++i -> step = 1
        if (UO->isIncrementOp()) {
          step = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, 1);
        } else if (UO->isDecrementOp()) {
          // i-- or --i -> step = -1
          step = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, -1);
        }
      } else if (const auto *BO = dyn_cast<BinaryOperator>(FS->getInc())) {
        // i += step or i = i + step
        if (BO->getOpcode() == BO_AddAssign || BO->getOpcode() == BO_Assign) {
          if (const auto *IL = dyn_cast<IntegerLiteral>(BO->getRHS()->IgnoreImpCasts())) {
            int64_t val = IL->getValue().getSExtValue();
            step = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, val);
          }
        }
      }
    }

    // Default step to 1 if not found
    if (!step) {
      step = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, 1);
    }
  } 

  // populate a struct that will be passed to emitForStmt (loop_nest)
  currentOMPLoopBounds = LoopBounds{lowerBound, upperBound, step, inclusive};

  // Create wsloop with empty parameters for now
  auto wsloopOp = builder.create<mlir::omp::WsloopOp>(
      scopeLoc,
      /*allocate_vars=*/mlir::ValueRange{},
      /*allocator_vars=*/mlir::ValueRange{},
      /*linear_vars=*/mlir::ValueRange{},
      /*linear_step_vars=*/mlir::ValueRange{},
      /*nowait=*/false,
      /*order=*/nullptr,
      /*order_mod=*/nullptr,
      /*ordered=*/nullptr,
      /*private_vars=*/mlir::ValueRange{},
      /*private_syms=*/nullptr,
      /*private_needs_barrier=*/false,
      /*reduction_mod=*/nullptr,
      /*reduction_vars=*/mlir::ValueRange{},
      /*reduction_byref=*/nullptr,
      /*reduction_syms=*/nullptr,
      /*schedule_kind=*/nullptr,
      /*schedule_chunk=*/nullptr,
      /*schedule_mod=*/nullptr,
      /*schedule_simd=*/false
  );


  // Populate the region with the ForStmt
  mlir::Region &region = wsloopOp.getRegion();
  mlir::Block *block = new mlir::Block();
  region.push_back(block);
  
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  if (emitStmt(FS, /*useCurrentScope=*/false).failed()) {
    res = mlir::failure();
  }

  // After emitStmt:
  currentOMPLoopBounds = std::nullopt; // Clear

  // omp.wsloop` does not require a yield or a terminator.
  return res;
}


//===----------------------------------------------------------------------===//
// emit OMP Parallel For Directive
//===----------------------------------------------------------------------===//

// lucap: trying to implement OMP Parallel For Directive
mlir::LogicalResult
CIRGenFunction::emitOMPParallelForDirective(
    const OMPParallelForDirective &S) {

  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  // 1. omp.parallel
  auto parallelOp = ParallelOp::create(builder, scopeLoc);
  mlir::Block &parallelBlock = parallelOp.getRegion().emplaceBlock();

  llvm::SmallVector<mlir::Value> lowerBounds, upperBounds, steps;   
    
  // Extract lower bound
  if (auto *lbExpr = S.getLowerBoundVariable()) {
    //llvm::errs() << "Found lower bound expression!\n" << lbExpr->dump() << "\n";
    lbExpr -> dump();
    auto lb = emitScalarExpr(lbExpr);
    lowerBounds.push_back(lb);
  }
    

  mlir::OpBuilder::InsertionGuard pg(builder);
  builder.setInsertionPointToEnd(&parallelBlock);

  // 2. CIR scope
  cir::ScopeOp::create(
      builder, scopeLoc,
      [&](mlir::OpBuilder &, mlir::Location) {

        LexicalScope lexScope{*this, scopeLoc,
                              builder.getInsertionBlock()};

        // 3. Allocate loop counter EARLY
        const VarDecl *loopVar = nullptr;
        std::optional<Address> loopVarAddr; // ✅ CORRECT

        auto counters = S.counters();
        if (!counters.empty()) {
          auto *counterRef = dyn_cast<DeclRefExpr>(counters[0]);
          auto *varDecl =
              counterRef ? dyn_cast<VarDecl>(counterRef->getDecl()) : nullptr;

          if (varDecl) {
            loopVar = varDecl;

            auto elemTy = convertType(varDecl->getType());
            auto alignment =
                getContext().getTypeAlignInChars(varDecl->getType());
            auto alignAttr =
                builder.getI64IntegerAttr(alignment.getQuantity());

            auto ptrTy =
                cir::PointerType::get(builder.getContext(), elemTy);

            auto alloca = cir::AllocaOp::create(
                builder,
                scopeLoc,
                ptrTy,
                elemTy,
                varDecl->getName(),
                alignAttr);

            loopVarAddr.emplace(alloca, elemTy, alignment);
            LocalDeclMap.insert({varDecl, *loopVarAddr});
          }
        }

        // 4. PreInits
        if (auto *preInits = cast_or_null<DeclStmt>(S.getPreInits())) {
          if (emitStmt(preInits, /*useCurrentScope=*/true).failed()) {
            res = mlir::failure();
            return;
          }
        }

        // 5. omp.wsloop
        auto wsloopOp = WsloopOp::create(builder, scopeLoc);
        mlir::Block *wsloopBlock =
            &wsloopOp.getRegion().emplaceBlock();
        builder.setInsertionPointToStart(wsloopBlock);

        // 6. Bounds
        llvm::SmallVector<mlir::Value> lbs, ubs, steps;

        if (auto *lb = S.getLowerBoundVariable())
          lbs.push_back(emitScalarExpr(lb));
        if (auto *ub = S.getUpperBoundVariable())
          ubs.push_back(emitScalarExpr(ub));
        if (auto *st = S.getStrideVariable())
          steps.push_back(emitScalarExpr(st));

        // 7. loop_nest
        auto loopNestOp = LoopNestOp::create(
            builder,
            scopeLoc,
            1,
            lbs,
            ubs,
            steps,
            /*inclusive=*/false,
            nullptr);

        mlir::Block *body =
            &loopNestOp.getRegion().emplaceBlock();
        body->addArgument(builder.getIndexType(), scopeLoc);

        // 8. Body
        builder.setInsertionPointToStart(body);

        mlir::Value iv = body->getArgument(0);

        if (loopVarAddr) {
          cir::StoreOp::create(
              builder,
              scopeLoc,
              iv,
              loopVarAddr->getPointer());
        }

        if (emitStmt(S.getStructuredBlock(),
                     /*useCurrentScope=*/true)
                .failed())
          res = mlir::failure();

        mlir::omp::YieldOp::create(
            builder, getLoc(S.getSourceRange().getEnd()));
      });

  // 9. terminator
  TerminatorOp::create(
      builder, getLoc(S.getSourceRange().getEnd()));

  return res;
}
