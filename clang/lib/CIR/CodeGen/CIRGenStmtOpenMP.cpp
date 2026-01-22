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

mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.barrier operation
  CGM.getOpenMPRuntime().emitBarrierCall(builder, *this,
                                         getLoc(S.getSourceRange()));
  return res;
}

// lucap: trying to implement OMP For Directive
mlir::LogicalResult
CIRGenFunction::emitOMPForDirective(const OMPForDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  // 4. Extract loop bounds from the AST
    llvm::SmallVector<mlir::Value> lowerBounds, upperBounds, steps;   


    
    // Extract upper bound
    if (auto *ubExpr = S.getUpperBoundVariable()) {
      auto ub = emitScalarExpr(ubExpr);
      upperBounds.push_back(ub);
    }
    
    // Extract stride
    if (auto *strideExpr = S.getStrideVariable()) {
      auto stride = emitScalarExpr(strideExpr);
      steps.push_back(stride);
    }

  // 1. Create the omp.wsloop operation
  auto wsloopOp = WsloopOp::create(builder, scopeLoc);
  
  // 2. Create a block inside wsloop's region and set insertion point there
  mlir::Block *wsloopBlock = &wsloopOp.getRegion().emplaceBlock();
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(wsloopBlock);
    
    // 3. CRITICAL: Emit the PreInits statement first
    if (auto *preInits = cast_or_null<DeclStmt>(S.getPreInits())) {
      if (emitStmt(preInits, /*useCurrentScope=*/true).failed()) {
        return mlir::failure();
      }
    }
    
    // 4. Extract loop bounds from the AST
    llvm::SmallVector<mlir::Value> lowerBounds, upperBounds, steps;   

    // Extract lower bound
    if (auto *lbExpr = S.getLowerBoundVariable()) {
      llvm::errs() << "Found lower bound expression!\n" << lbExpr->getStmtClassName() << "\n";
      //lbExpr -> dump();
      auto lb = emitScalarExpr(lbExpr);
      lowerBounds.push_back(lb);
    }
    
    // Extract upper bound
    if (auto *ubExpr = S.getUpperBoundVariable()) {
      auto ub = emitScalarExpr(ubExpr);
      upperBounds.push_back(ub);
    }
    
    // Extract stride
    if (auto *strideExpr = S.getStrideVariable()) {
      auto stride = emitScalarExpr(strideExpr);
      steps.push_back(stride);
    }
    
    // 5. Create the omp.loop_nest operation
    auto loopNestOp = LoopNestOp::create(
        builder, 
        scopeLoc,
        /*collapse_num_loops=*/ 1,
        /*loop_lower_bounds=*/ lowerBounds,
        /*loop_upper_bounds=*/ upperBounds,
        /*loop_steps=*/ steps,
        /*loop_inclusive=*/ false,
        /*tile_sizes=*/ nullptr
    );
    
    // 6. Create a block inside loop_nest's region with the induction variable
    mlir::Block *loopBodyBlock = &loopNestOp.getRegion().emplaceBlock();
    auto indexTy = builder.getIndexType();
    loopBodyBlock->addArgument(indexTy, scopeLoc);
    
    {
      mlir::OpBuilder::InsertionGuard loopGuard(builder);
      builder.setInsertionPointToStart(loopBodyBlock);
      
      // 7. Map the loop counter to the induction variable
      mlir::Value inductionVar = loopBodyBlock->getArgument(0);

      auto counters = S.counters();
      if (!counters.empty()) {
        if (auto *counterRef = dyn_cast<DeclRefExpr>(counters[0])) {
          if (auto *varDecl = dyn_cast<VarDecl>(counterRef->getDecl())) {
            
            // Convert the Clang type to CIR type
            auto counterType = convertType(varDecl->getType());
            auto alignment = getContext().getTypeAlignInChars(varDecl->getType());
            
            // Convert alignment to IntegerAttr
            auto alignmentAttr = builder.getI64IntegerAttr(alignment.getQuantity());
            
            // Compute pointer type
            auto ptrType = cir::PointerType::get(builder.getContext(), counterType);

            // Create an alloca for the counter
            // Signature: create(OpBuilder &builder, Location location, Type addr, Type allocaType, StringRef name, IntegerAttr alignment)
            auto alloca = cir::AllocaOp::create(
                builder,
                scopeLoc,
                ptrType,           // addr (result type)
                counterType,       // allocaType
                varDecl->getName(),// name
                alignmentAttr);    // alignment
            
            // Store the induction variable into the alloca
            builder.create<cir::StoreOp>(scopeLoc, inductionVar, alloca);
            
            // Create an Address object and insert it into LocalDeclMap
            Address counterAddr(alloca, counterType, alignment);
            LocalDeclMap.insert({varDecl, counterAddr});
          }
        }
      }
      
      // 8. Emit the loop body
      if (emitStmt(S.getStructuredBlock(), /*useCurrentScope=*/true).failed()) {
        res = mlir::failure();
      }
      
      // 9. Add omp.yield
      mlir::omp::YieldOp::create(builder, getLoc(S.getSourceRange().getEnd()));
    }
  }
  
  return res;
}



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
