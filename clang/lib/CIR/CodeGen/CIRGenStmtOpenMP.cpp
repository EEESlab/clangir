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
// Emit OpenMP `omp.for` directive
//
// This function lowers a Clang `OMPForDirective` into an MLIR OpenMP
// `omp.wsloop` operation. The loop body and iteration space are emitted
// separately by visiting the associated `ForStmt`.
//
// Design note:
//  - This function is responsible only for creating the OpenMP worksharing
//    construct and extracting loop bounds.
//  - The actual loop nest (`omp.loop_nest`) is emitted later when the
//    `ForStmt` is visited, using the bounds computed here.
//===----------------------------------------------------------------------===//



mlir::LogicalResult   
CIRGenFunction::emitOMPForDirective(const OMPForDirective &S) {   

  // Assume success unless an error is encountered while emitting the body.
  mlir::LogicalResult res = mlir::success();

  // Source location used for all operations created for this directive.
  auto scopeLoc = getLoc(S.getSourceRange());
  llvm::errs() << "DEBUG: creating omp.wsloop op\n";

  // OpenMP `for` directives wrap the associated loop inside a CapturedStmt.
  // Extract the underlying canonical `for` loop.
  const CapturedStmt *capturedStmt = S.getInnermostCapturedStmt();
  const ForStmt *forStmt = dyn_cast<ForStmt>(capturedStmt->getCapturedStmt());

  // Loop bounds extracted from the Clang AST.
  //
  // IMPORTANT:
  // These values are materialized *outside* of the `omp.wsloop` region so
  // that they dominate the loop nest emitted later. This matches the
  // expectations of the OpenMP dialect, where loop bounds are SSA values
  // available to the loop_nest.
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool inclusive = false; // true for <= or >= loop conditions
  
  if (forStmt) {
    //===------------------------------------------------------------------===//
    // 1. Lower bound
    //
    // Handles canonical loop initializers of the form:
    //   for (int i = <init>; ...)
    //
    // Non-canonical forms are currently not supported, as the OpenMP
    // `loop_nest` operation expects a normalized loop structure.
    //===------------------------------------------------------------------===//
    if (const auto *declStmt = dyn_cast<DeclStmt>(forStmt->getInit())) {
      if (const auto *varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl())) {
        if (varDecl->hasInit()) {
          // Try constant first
          if (const auto *intLit = dyn_cast<IntegerLiteral>(varDecl->getInit()->IgnoreImpCasts())) {
            lowerBound = mlir::arith::ConstantIndexOp::create(builder, scopeLoc, intLit->getValue().getSExtValue());
          } else {   
            // For non-constant bounds, emit as CIR value
            mlir::Value rawLB = emitScalarExpr(varDecl->getInit());
            // CIR expressions typically produce cir.int types, but OpenMP loop bounds must be of type `index`.
            auto cirIntType = mlir::dyn_cast<cir::IntType>(rawLB.getType());
            // Convert cir.int -> builtin integer (i32/i64).
            mlir::Type stdIntTy = builder.getIntegerType(cirIntType.getWidth());
            
            // UnrealizedConversionCast is used here as a temporary bridge between CIR types and standard MLIR types.
            auto castOpLB =
                mlir::UnrealizedConversionCastOp::create(
                    builder, scopeLoc,
                    mlir::TypeRange{stdIntTy},
                    mlir::ValueRange{rawLB});

            // Convert builtin integer -> index.
            lowerBound = mlir::arith::IndexCastOp::create(
                builder, scopeLoc,
                builder.getIndexType(),
                castOpLB.getResult(0));
          }
        }
      }
    }


    //===------------------------------------------------------------------===//
    // 2. Upper bound and comparison kind
    //
    // Handles loop conditions of the form:
    //   i < N, i <= N, i > N, i >= N
    //
    // The RHS expression defines the bound, while the comparison operator
    // determines whether the bound is inclusive.
    //===------------------------------------------------------------------===//
    if (forStmt->getCond()) {
    if (const auto *binOp = dyn_cast<BinaryOperator>(forStmt->getCond())) {
        // Try constant first
        if (const auto *intLit = dyn_cast<IntegerLiteral>(binOp->getRHS()->IgnoreImpCasts())) {
            int64_t boundVal = intLit->getValue().getSExtValue();
            upperBound = mlir::arith::ConstantIndexOp::create(builder, scopeLoc, boundVal);
        } else {
            // For non-constant bounds, emit as CIR value
            mlir::Value rawBound = emitScalarExpr(binOp->getRHS());
            auto cirIntType = mlir::dyn_cast<cir::IntType>(rawBound.getType());
            mlir::Type stdIntTy = builder.getIntegerType(cirIntType.getWidth());
            auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
                scopeLoc, stdIntTy, rawBound);
            upperBound = mlir::arith::IndexCastOp::create(
                builder, scopeLoc, builder.getIndexType(), castOp.getResult(0));
        }
        BinaryOperatorKind opKind = binOp->getOpcode();
        inclusive = (opKind == BO_LE || opKind == BO_GE);
    }
}


    //===------------------------------------------------------------------===//
    // 3. Step
    //
    // Supports the following increment forms:
    //   i++, ++i, i--, --i
    //   i += step
    //   i = i + step
    //===------------------------------------------------------------------===//
    if (forStmt->getInc()) {
      if (const auto *unaryOp = dyn_cast<UnaryOperator>(forStmt->getInc())) {
        int64_t val = unaryOp->isIncrementOp() ? 1 : -1;
        step = mlir::arith::ConstantIndexOp::create(builder, scopeLoc, val);
      } else if (const auto *binOp = dyn_cast<BinaryOperator>(forStmt->getInc())) {
        Expr *stepExpr = nullptr;
        if (binOp->isCompoundAssignmentOp()) {
          stepExpr = binOp->getRHS();
        } else if (binOp->isAssignmentOp()) {
          if (auto *subBinOp = dyn_cast<BinaryOperator>(binOp->getRHS()->IgnoreImpCasts())) {
            stepExpr = subBinOp->getRHS();
          }
        }

        if (stepExpr) {
          // Try constant first
          if (const auto *intLit = dyn_cast<IntegerLiteral>(stepExpr->IgnoreImpCasts())) {
            step = mlir::arith::ConstantIndexOp::create(builder, scopeLoc, intLit->getValue().getSExtValue());
          } else {
            mlir::Value rawStep = emitScalarExpr(stepExpr);
            auto cirIntType = mlir::dyn_cast<cir::IntType>(rawStep.getType());
            mlir::Type stdIntTy = builder.getIntegerType(cirIntType.getWidth());

            auto castOpStep =
                mlir::UnrealizedConversionCastOp::create(
                    builder, scopeLoc,
                    mlir::TypeRange{stdIntTy},
                    mlir::ValueRange{rawStep});

            step = mlir::arith::IndexCastOp::create(
                builder, scopeLoc,
                builder.getIndexType(),
                castOpStep.getResult(0));
          }
        }
      }
    }

    // Default to a unit step if no increment expression was recognized.
    if (!step) {
      step = mlir::arith::ConstantIndexOp::create(builder, scopeLoc, 1);
    }
  }

  // Store the extracted bounds so that `emitForStmt` can construct the
  // corresponding `omp.loop_nest` inside the wsloop region.
  currentOMPLoopBounds = LoopBounds{lowerBound, upperBound, step, inclusive};

  // Create the OpenMP worksharing loop operation.
  // Most clauses are currently unimplemented and left empty.
  auto wsloopOp = mlir::omp::WsloopOp::create(
      builder, scopeLoc,
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


  // Populate the wsloop region by emitting the associated `for` statement.
  mlir::Region &region = wsloopOp.getRegion();
  mlir::Block *block = new mlir::Block();
  region.push_back(block);
  
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  if (emitStmt(forStmt, /*useCurrentScope=*/false).failed()) {
    res = mlir::failure();
  }

  // Clear loop-bound state after emitting the loop body.
  currentOMPLoopBounds = std::nullopt; // Clear

  // `omp.wsloop` does not require an explicit terminator or yield.
  return res;
}