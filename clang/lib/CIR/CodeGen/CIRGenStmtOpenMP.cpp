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

// lucap: implementing OMP For Directive
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
    // === 1. Handle Lower Bound (e.g., int i = start) ===
    if (const auto *DS = dyn_cast<DeclStmt>(FS->getInit())) {
      if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
        if (VD->hasInit()) {
          mlir::Value rawLB = emitScalarExpr(VD->getInit());
          if (rawLB) {
            auto cirIntType = mlir::dyn_cast<cir::IntType>(rawLB.getType());
            if (cirIntType) {
              mlir::Type stdIntTy = builder.getIntegerType(cirIntType.getWidth());
              mlir::Value stdInt = builder.create<mlir::UnrealizedConversionCastOp>(
                  scopeLoc, stdIntTy, rawLB).getResult(0);
              lowerBound = mlir::arith::IndexCastOp::create(
                  builder, scopeLoc, builder.getIndexType(), stdInt);
            }
          }
        }
      }
    }


    // Handle Upper Bound and inclusive
    if (FS->getCond()) {
        if (const auto *BO = dyn_cast<BinaryOperator>(FS->getCond())) {
          // 1. Evaluate the RHS (this could be 'b' or '10')
          mlir::Value rawBound = emitScalarExpr(BO->getRHS());
          // 2. CONVERT cir.int to mlir.index (which OpenMP dialect expects)
          // You likely need to cast the CIR type to a standard MLIR type
          if (rawBound) {
            // 1. Correct way to cast Type in modern MLIR
            auto cirIntType = mlir::dyn_cast<cir::IntType>(rawBound.getType());
            if (cirIntType) {
              // 1. Get the standard MLIR integer type (i32/i64)
              mlir::Type stdIntTy = builder.getIntegerType(cirIntType.getWidth());
              
              // use Unrealized conversion cast to force the cast when type conflict
              mlir::Value stdInt = builder.create<mlir::UnrealizedConversionCastOp>(
                  scopeLoc, stdIntTy, rawBound).getResult(0);

              // 3. Now convert standard i32 to index
              upperBound = mlir::arith::IndexCastOp::create(
                  builder, scopeLoc, builder.getIndexType(), stdInt);
            }
          }
          // Check if comparison is inclusive (<= or >=) or exclusive (< or >)
          BinaryOperatorKind opKind = BO->getOpcode();
          if (opKind == BO_LE || opKind == BO_GE) {
            inclusive = true;
          }
        }
      }


    // === 3. Handle Step (e.g., i++, i += step) ===
    if (FS->getInc()) {
      if (const auto *UO = dyn_cast<UnaryOperator>(FS->getInc())) {
        int64_t val = UO->isIncrementOp() ? 1 : -1;
        step = mlir::arith::ConstantIndexOp::create(builder, scopeLoc, val);
      } else if (const auto *BO = dyn_cast<BinaryOperator>(FS->getInc())) {
        // Support i += step OR i = i + step
        Expr *stepExpr = nullptr;
        if (BO->isCompoundAssignmentOp()) {
          stepExpr = BO->getRHS();
        } else if (BO->isAssignmentOp()) {
          if (auto *SubBO = dyn_cast<BinaryOperator>(BO->getRHS()->IgnoreImpCasts())) {
            stepExpr = SubBO->getRHS();
          }
        }

        if (stepExpr) {
          mlir::Value rawStep = emitScalarExpr(stepExpr);
          if (rawStep) {
            auto cirIntType = mlir::dyn_cast<cir::IntType>(rawStep.getType());
            if (cirIntType) {
              mlir::Type stdIntTy = builder.getIntegerType(cirIntType.getWidth());
              mlir::Value stdInt = builder.create<mlir::UnrealizedConversionCastOp>(
                  scopeLoc, stdIntTy, rawStep).getResult(0);
              step = mlir::arith::IndexCastOp::create(
                  builder, scopeLoc, builder.getIndexType(), stdInt);
            }
          }
        }
      }
    }

    // Default step to 1 if not found
    if (!step) {
      step = builder.create<mlir::arith::ConstantIndexOp>(scopeLoc, 1);
    }
  } 

  llvm::errs() << "=== Generated wsloop operation ===\n";
  upperBound.dump();
  llvm::errs() << "=== End of wsloop ===\n";

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