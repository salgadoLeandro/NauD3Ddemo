
RaytracingAccelerationStructure gRtScene : register(t0);

RWTexture2D<float4> output0 : register(u0);

ByteAddressBuffer position[45] : register(t100);
ByteAddressBuffer normal[45] : register(t200);
ByteAddressBuffer texCoord0[45] : register(t300);
ByteAddressBuffer index[45] : register(t400);

Texture2D<float4> tex[45] : register(t500);

RWTexture2D<float4> gPos : register(u1);
RWTexture2D<float4> gNorm : register(u2);

cbuffer Camera : register(b0) {
    float4 eye;
    float4 V;
    float4 U;
    float4 W;
    float fov;
};

cbuffer GlobalAttributes : register(b1) {
    float4 lightDir;
    float4 lightPos;
    uint frameCount;
};

struct MaterialAttrs {
    float4 diffuse;
    int texCount;
};

cbuffer MaterialAttributes : register(b2) {
    MaterialAttrs materialAttributes[45];
};

struct RayPayload {
    float aoValue;
};


uint initRand (uint val0, uint val1, uint backoff = 16) {
    uint v0 = val0, v1 = val1, s0 = 0;

    [unroll]
    for (uint n = 0; n < backoff; n++) {

        s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

float nextRand(inout uint s) {
    s = (1664525u * s + 1013904223u);
	return float(s & 0x00FFFFFF) / float(0x01000000);
}


float3 getPerpendicularVector(float3 u) {

    float3 a = abs(u);
    uint xm = ((a.x - a.y) < 0 && (a.x - a.z) < 0) ? 1 : 0;
    uint ym = (a.y - a.z) < 0 ? (1 ^ xm) : 0;
    uint zm = 1 ^ (xm | ym);
    return cross(u, float3(xm, ym, zm));
}


float3 getCosHemisphereSample(inout uint randSeed, float3 hitNorm) {

    float2 randVal = float2(nextRand(randSeed), nextRand(randSeed));

    float3 bitangent = getPerpendicularVector(hitNorm);
    float3 tangent = cross(bitangent, hitNorm);
    float r = sqrt(randVal.x);
    float phi = 2.0f * 3.14159265f * randVal.y;

    return tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + hitNorm.xyz * sqrt(1 - randVal.x);
}


float shootAmbientOcclusionRay(float3 orig, float3 dir, float minT, float maxT) {

    RayPayload payload = { 0.0f };
    
    RayDesc ray;
    ray.Origin = orig;
    ray.Direction = dir;
    ray.TMin = minT;
    ray.TMax = maxT;

    uint flag = 0x04 | 0x08; //RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
    TraceRay(gRtScene, flag, 0xFF, 0, 2, 0, ray, payload);

    return payload.aoValue;
}

#define mod(x, y) (x - (y * int(x / y)))

static const float gAORadius = 1000.0f;
static const float gMinT = 0.0001f;
static const uint gNumRays = 4;


[shader("raygeneration")]
void raygen() {

    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    uint randSeed = initRand(launchIndex.x + launchIndex.y * launchDim.x, frameCount);

    float4 worldPos = gPos[launchIndex];
    float4 worldNor = gNorm[launchIndex];

    float ambientOcclusion = float(gNumRays);

    ambientOcclusion = 0.0f;
    
    for(int i = 0; i < gNumRays; ++i) {
        float3 worldDir = getCosHemisphereSample(randSeed, worldNor.xyz);

        ambientOcclusion += shootAmbientOcclusionRay(worldPos.xyz, worldDir, gMinT, gAORadius);
    }

    float aoColor = ambientOcclusion / float(gNumRays);
    output0[launchIndex] = float4(aoColor, aoColor, aoColor, 1.0f);
}


[shader("miss")]
void aoMiss(inout RayPayload payload) {

    payload.aoValue = 1.0f;
}


[shader("anyhit")]
void aoAnyHit(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs) {

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint tIndex = PrimitiveIndex();
    int address = tIndex * 3 * 4;
    uint3 indices = index[InstanceID()].Load3(address);

    float2 texCoord = float2(0.0f, 0.0f);
    for(uint i = 0; i < 3; ++i) {
        uint vaddr = indices[i] * 4 * 4;
        uint4 texc = texCoord0[InstanceID()].Load4(vaddr);
        texCoord += asfloat(texc.xy) * barycentrics[i];
    }
    uint2 tsize;
    tex[InstanceID()].GetDimensions(tsize.x, tsize.y);

    texCoord = float2(mod(texCoord.x, 1.0f), mod(texCoord.y, 1.0f));
    if (texCoord.x < 0) texCoord.x = 1 + texCoord.x;
    if (texCoord.y < 0) texCoord.y = 1 + texCoord.y;

    if (tex[InstanceID()][uint2(texCoord * tsize)].w < 0.25f) {
        IgnoreHit();
    }
}