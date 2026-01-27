// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func
void omp_for_1() {
// CHECK: omp.parallel {
// CHECK-NEXT: cir.scope {
// CHECK: %{{.+}} = cir.const #cir.int<0> : !s32i
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[LB:.+]] = arith.index_cast
// CHECK: %{{.+}} = cir.const #cir.int<10> : !s32i
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[UB:.+]] = arith.index_cast
// CHECK: %[[STEP:.+]] = arith.constant 1 : index
// CHECK: omp.wsloop {
// CHECK-NEXT: omp.loop_nest (%{{.+}}) : index = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK: omp.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
    #pragma omp parallel 
    { 
        #pragma omp for
        for (int i = 0; i < 10; ++i) {
        }
    }
}

// CHECK: cir.func
void omp_for_2(int a) {
// CHECK: %[[A:.+]] = {{.*}} ["a", init]
// CHECK: %[[SUM:.+]] = {{.*}} ["sum", init]
// CHECK: %[[B:.+]] = {{.*}} ["b", init]
// CHECK: %[[C:.+]] = {{.*}} ["c", init]
// CHECK: omp.parallel {
// CHECK-NEXT: cir.scope {
// CHECK: %{{.+}} = cir.load {{.*}} %[[A]]
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[LB:.+]] = arith.index_cast
// CHECK: %{{.+}} = cir.load {{.*}} %[[B]]
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[UB:.+]] = arith.index_cast
// CHECK: %{{.+}} = cir.load {{.*}} %[[C]]
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[STEP:.+]] = arith.index_cast
// CHECK: omp.wsloop {
// CHECK-NEXT: omp.loop_nest (%{{.+}}) : index = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK: cir.scope {
// CHECK: %{{.+}} = cir.load {{.*}} %[[SUM]]
// CHECK: %{{.+}} = cir.cast{{.*}}!s64i
// CHECK: %{{.+}} = cir.load {{.*}} %[[B]]
// CHECK: %{{.+}} = cir.binop(add
// CHECK: %{{.+}} = cir.cast{{.*}}!s32i
// CHECK: cir.store {{.*}} %[[SUM]]
// CHECK-NEXT: }
// CHECK: omp.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
    int sum = 0;    
    long long b = 10;
    unsigned c = 2;
    
    #pragma omp parallel
    {
        #pragma omp for
        for (int i=a; i < b; i=i+c) {
            sum = sum + b;
        }
    }
}


// CHECK: cir.func
void omp_for_3() {
// CHECK: omp.parallel {
// CHECK-NEXT: cir.scope {
// CHECK: %{{.+}} = cir.const #cir.int<0> : !s32i
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[LB:.+]] = arith.index_cast
// CHECK: %{{.+}} = cir.const #cir.int<10> : !s32i
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[UB:.+]] = arith.index_cast
// CHECK: %{{.+}} = cir.const #cir.int<1> : !s32i
// CHECK: %{{.+}} = builtin.unrealized_conversion_cast
// CHECK: %[[STEP:.+]] = arith.index_cast
// CHECK: omp.wsloop {
// CHECK-NEXT: omp.loop_nest (%{{.+}}) : index = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK: cir.scope {
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[J:.+]] = {{.*}} ["j", init]
// CHECK: cir.for : cond {
// CHECK: %{{.+}} = cir.load {{.*}} %[[J]]
// CHECK: %{{.+}} = cir.const #cir.int<3> : !s32i
// CHECK: %{{.+}} = cir.cmp(lt
// CHECK: cir.condition
// CHECK-NEXT: } body {
// CHECK: cir.yield
// CHECK-NEXT: } step {
// CHECK: %{{.+}} = cir.load {{.*}} %[[J]]
// CHECK: %{{.+}} = cir.const #cir.int<1> : !s32i
// CHECK: %{{.+}} = cir.binop(add
// CHECK: cir.store {{.*}} %[[J]]
// CHECK: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK: omp.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
    #pragma omp parallel
    {
        #pragma omp for
        for (int i=0; i < 10; i=i+1) {
            for (int j=0; j < 3; j=j+1){
            }
        }
    }
}