// ANE MIL op-set prober.
// dlopens AppleNeuralEngine.framework, builds an in-memory model from MIL text,
// compiles it, and reports whether the ANE runtime accepted the program.
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <string>
#include <vector>
#include <cstdio>

typedef id (*msg0)(id, SEL);
typedef id (*msg1)(id, SEL, id);
typedef id (*msg3)(id, SEL, id, id, id);
typedef BOOL (*msgQoS)(id, SEL, long, id, id*);

static Class C_DESC, C_MODEL;

static bool compileMIL(NSString *mil, NSString **errOut, long qos) {
  @autoreleasepool {
    SEL sDesc = sel_registerName("modelWithMILText:weights:optionsPlist:");
    if (![C_DESC respondsToSelector:sDesc]) { *errOut = @"selector missing on descriptor"; return false; }

    NSDictionary *weights;
    const char *shape = getenv("WSHAPE") ?: "dict";
    if (!strcmp(shape, "dict"))       weights = @{@"w": @{}};
    else if (!strcmp(shape, "array")) weights = @{@"w": @[]};
    else if (!strcmp(shape, "arrdata")) weights = @{@"w": @[[NSData data]]};
    else if (!strcmp(shape, "empty")) weights = @{};
    else weights = @{@"w": [NSData data]};
    __unsafe_unretained NSString **pIdent  = (__unsafe_unretained NSString**)dlsym(RTLD_DEFAULT, "kANEFInMemoryModelIdentifierKey");
    __unsafe_unretained NSString **pCached = (__unsafe_unretained NSString**)dlsym(RTLD_DEFAULT, "kANEFInMemoryModelIsCachedKey");
    NSString *kIdent  = pIdent  ? *pIdent  : @"kANEFInMemoryModelIdentifierKey";
    NSString *kCached = pCached ? *pCached : @"kANEFInMemoryModelIsCachedKey";
    printf("  keys: identifier=%s cached=%s\n", [kIdent UTF8String], [kCached UTF8String]);
    NSDictionary *optsDict = @{ kIdent: [[NSUUID UUID] UUIDString], kCached: @NO };
    NSData *opts = [NSPropertyListSerialization dataWithPropertyList:optsDict
                     format:NSPropertyListBinaryFormat_v1_0 options:0 error:nil];
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((msg3)objc_msgSend)((id)C_DESC, sDesc, (id)milData, (id)weights, (id)opts);
    if (!desc) { *errOut = @"descriptor nil"; return false; }

    SEL sModel = sel_registerName("inMemoryModelWithDescriptor:");
    id model = ((msg1)objc_msgSend)((id)C_MODEL, sModel, desc);
    if (!model) { *errOut = @"model nil"; return false; }

    SEL sCompile = sel_registerName("compileWithQoS:options:error:");
    NSError *err = nil;
    BOOL ok = ((msgQoS)objc_msgSend)(model, sCompile, qos, @{}, (id*)&err);
    if (!ok) {
      *errOut = err ? [err localizedDescription] : @"compile failed, no NSError";
      if (err && err.userInfo.count)
        *errOut = [NSString stringWithFormat:@"%@ | %@", *errOut, err.userInfo];
      return false;
    }
    return true;
  }
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    setbuf(stdout, NULL);
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    C_DESC  = objc_getClass("_ANEInMemoryModelDescriptor");
    C_MODEL = objc_getClass("_ANEInMemoryModel");
    printf("framework loaded. desc=%p model=%p\n", (__bridge void*)C_DESC, (__bridge void*)C_MODEL);
    if (!C_DESC || !C_MODEL) return 1;

    // read MIL from stdin
    NSFileHandle *in = [NSFileHandle fileHandleWithStandardInput];
    NSData *d = [in readDataToEndOfFile];
    NSString *mil = [[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding];

    long qos = (argc > 1) ? atol(argv[1]) : 0;
    NSString *err = nil; bool ok = false;
    @try { ok = compileMIL(mil, &err, qos); }
    @catch (NSException *e) { printf("EXC: %s | %s\n", [[e name] UTF8String], [[e reason] UTF8String]); return 3; }
    printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    if (!ok) printf("ERROR: %s\n", err ? [err UTF8String] : "(none)");
    return ok ? 0 : 2;
  }
}
