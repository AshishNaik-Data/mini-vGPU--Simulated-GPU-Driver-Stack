/* test_framework.h - a deliberately tiny test harness: no dependencies. */
#ifndef VGPU_TEST_FRAMEWORK_H
#define VGPU_TEST_FRAMEWORK_H

#include <stdio.h>

static int g_checks, g_failures;

#define CHECK(cond)                                                              \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            g_failures++;                                                        \
            fprintf(stderr, "\n  %s:%d: CHECK failed: %s", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

#define CHECK_EQ(actual, expected)                                               \
    do {                                                                         \
        long long a_ = (long long)(actual), e_ = (long long)(expected);          \
        g_checks++;                                                              \
        if (a_ != e_) {                                                          \
            g_failures++;                                                        \
            fprintf(stderr, "\n  %s:%d: %s == %lld, expected %lld", __FILE__, __LINE__, \
                    #actual, a_, e_);                                            \
        }                                                                        \
    } while (0)

#define RUN(test)                                                                \
    do {                                                                         \
        int before_ = g_failures;                                                \
        printf("  %-44s", #test);                                                \
        fflush(stdout);                                                          \
        test();                                                                  \
        printf("%s\n", g_failures == before_ ? "ok" : "  <-- FAIL");             \
    } while (0)

#define TEST_SUMMARY()                                                           \
    (printf("\n%d checks, %d failed\n", g_checks, g_failures), g_failures ? 1 : 0)

#endif /* VGPU_TEST_FRAMEWORK_H */
