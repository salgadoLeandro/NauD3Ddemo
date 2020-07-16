
#define mod(x, y) (x - (y * int(x / y)))

RaytracingAccelerationStructure gRtScene : register(t0);

RWTexture2D<float4> output0 : register(u0);

ByteAddressBuffer position[45] : register(t100);
ByteAddressBuffer normal[45] : register(t200);
ByteAddressBuffer texCoord0[45] : register(t300);
ByteAddressBuffer index[45] : register(t400);

Texture2D<float4> tex[45] : register(t500);

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
    float4 result;
    int depth;
    uint seed;
    float entrance;
};


static const float M_PI = 3.14159265f;
static const float2 lightSize = float2(0.4, 0.45);
static const uint lightSamples = 4;

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

float4 sampleAreaLight(float3 surfaceNormal, float3 hitPoint, float3 lightP,
                float3 lightN, float2 lightSize, inout uint seed) {
    
    RayPayload shadow_prd;
    shadow_prd.result = float4(0, 0, 0, 0);
    float4 result = float4(0, 0, 0, 0);

    float2 r;
    float dot1 = dot(hitPoint - lightP, lightN);
    if (dot1 < 0) {
        return result;
    }

    for(uint i = 0; i < lightSamples; ++i) {

        r.x = nextRand(seed);
        r.y = nextRand(seed);

        float3 lPos = lightP + float3(1, 0, 0) * lightSize.x * r.x + float3(0, 0, 1) * lightSize.y * r.y;
        float3 lDir = lPos - hitPoint;
        float lightDist = length(lDir);
        lDir = normalize(lDir);

        float NdotL = dot(surfaceNormal, lDir);
        if(NdotL > 0) {
            dot1 = max(0.0f, dot(lightN, -lDir));

            shadow_prd.result = float4(1, 1, 1, 1);

            RayDesc ray;
            ray.Origin = hitPoint;
            ray.Direction = lDir;
            ray.TMin = 0.001f;
            ray.TMax = lightDist + 0.01f;

            TraceRay(gRtScene, 0, 0xFF, 1, 0, 1, ray, shadow_prd);
            result += shadow_prd.result * NdotL * dot1;
        }
    }

    result.w = lightSamples;

    return (result / lightSamples);
}


[shader("raygeneration")]
void raygen() {
    uint3 launchIndex = DispatchRaysIndex();
	uint3 launchDim = DispatchRaysDimensions();

    uint2 screen;
    output0.GetDimensions(screen.x, screen.y);
    uint seed = initRand(screen.x * launchIndex.x + launchIndex.y, frameCount);
	
    float2 crd = float2(launchIndex.xy);
	float2 dims = float2(launchDim.xy);

	float2 d = ((crd / dims) * 2.0f - 1.0f);
	float aspectRatio = dims.x / dims.y;
	d.y = -d.y;

	float3 u = float3(U.xyz);
	float3 v = float3(V.xyz);
	float3 w = float3(W.xyz);

	RayDesc ray;
	ray.Origin = eye.xyz;
	ray.Direction = normalize(d.x*u*fov + d.y*v*fov + w);
	ray.TMin = 0;
	ray.TMax = 100000;

	RayPayload payload;
    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
    payload.depth = 0;
    payload.seed = seed;
	TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);

	output0[launchIndex.xy] = payload.result;
}


[shader("miss")]
void miss(inout RayPayload payload) {

    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("closesthit")]
void tracePath(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    if (payload.depth >= 4) return;

    RayPayload rp;
    rp.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 newDir;

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint tIndex = PrimitiveIndex();
    int address = tIndex * 3 * 4;
    uint3 indices = index[InstanceID()].Load3(address);

    float3 an = float3(0.0, 0.0, 0.0);
    for(uint i = 0; i < 3; ++i) {
        uint vaddr = indices[i] * 4 * 4;
        uint4 tan = normal[InstanceID()].Load4(vaddr);
        an += asfloat(tan.xyz) * barycentrics[i];
    }

    float3 n = normalize(an);
    float3 hit_point = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    float4 shadow = sampleAreaLight(n, hit_point, float3(lightPos.xyz), float3(0, -1, 0), lightSize, payload.seed);

    payload.result *= shadow;
}


[shader("closesthit")]
void shadeLight(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("miss")]
void missShadow(inout RayPayload payload) {

    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("anyhit")]
void any_hit_shadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
    payload.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
}


[shader("anyhit")]
void alpha_test(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

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


[shader("anyhit")]
void alpha_test_shadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

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
    else {
        payload.result = float4(0,0,0,0);
        AcceptHitAndEndSearch();
    }
}


[shader("closesthit")]
void traceGlass(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    if (payload.depth >= 4) return;

    float3 hit_point = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    float3 newDir = WorldRayDirection();

    RayDesc ray;
    ray.Origin = hit_point;
    ray.Direction = newDir;
    ray.TMin = 0.0001;
    ray.TMax = 5000;
    TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);

    payload.result *= 0.9f;
}


[shader("anyhit")]
void keepGoingShadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint tIndex = PrimitiveIndex();
    int address = tIndex * 3 * 4;
    uint3 indices = index[InstanceID()].Load3(address);

    float3 an = float3(0.0, 0.0, 0.0);
    for(uint i = 0; i < 3; ++i) {
        uint vaddr = indices[i] * 4 * 4;
        uint4 tan = normal[InstanceID()].Load4(vaddr);
        an += asfloat(tan.xyz) * barycentrics[i];
    }

    float3 n = normalize(an);
    float atenuation = 1.0f;

    if(payload.entrance == 0.0) {
        atenuation *= sqrt(abs(dot(n, WorldRayDirection())));
        payload.entrance = RayTCurrent();
    }
    else {
        atenuation = pow(exp(log(0.84) * (abs(RayTCurrent() - payload.entrance))), 4);
        atenuation *= sqrt(abs(dot(n, WorldRayDirection())));
    }
    payload.result *= atenuation;

    IgnoreHit();
}