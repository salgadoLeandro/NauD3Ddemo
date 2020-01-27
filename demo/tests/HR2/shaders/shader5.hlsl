#define mod(x, y) (x - (y * int(x / y)))

RaytracingAccelerationStructure gRtScene : register(t0);

RWTexture2D<float4> output0 : register(u0);
RWTexture2D<float> output1 : register(u1);
RWTexture2D<float4> output2 : register(u2);

RWTexture2D<float4> mask : register(u3);

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
    float4 shadow;
    float aoValue;
    int depth;
    uint seed;
};

static const float M_PI = 3.14159265f;

static const float AtG = 0.6668f;
static const float FresnelPower = 0.9f;
static const float F = ((1.0-AtG) * (1.0-AtG)) / ((1.0+AtG) * (1.0+AtG));

static const uint aoSamples = 4;
static const uint lightSamples = 1;
static const float2 lightSize = float2(0.4, 0.45);

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

void createONB(const float3 n, out float3 U, out float3 V) {

    U = cross(n, float3(0, 1, 0));
    if (dot(U, U) < 1.e-3f) {
        U = cross(n, float3(1, 0, 0));
    }
    U = normalize(U);
    V = cross(U, n);
}

float3 sampleUnitHemisphere(const float2 _sample, const float3 U, const float3 V, const float3 W) {

    float phi = 2.0f * M_PI * _sample.x;
    float r = sqrt(_sample.y);
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = 1.0f - x * x - y * y;
    z = z > 0.0f ? sqrt(z) : 0.0f;

    return x*U + y*V + z*W;
}

float3 sampleUnitHemisphereCosWeighted(float3 normal, inout uint seed) {

    float2 r = float2(nextRand(seed), nextRand(seed));
    float3 U, V;

    createONB(normal, U, V);
    return sampleUnitHemisphere(r, V, U, normal);
}

#define MS 1

[shader("raygeneration")]
void raygen() {
    
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

#if MS == 0
    float4 result = float4(0, 0, 0, 0);
    float4 shadow = float4(0, 0, 0, 0);
    float aoValue = 0.0f;
    float depthM = 0.0f;
    int sqrt_num_samples = 2;
    int samples = sqrt_num_samples * sqrt_num_samples;

    uint2 screen;
    output0.GetDimensions(screen.x, screen.y);
    
    uint seed = initRand(screen.x * launchIndex.x + launchIndex.y, frameCount);

    float2 inv_screen = 1.0f / float2(screen) * 2.0f;
    float2 pixel = float2(launchIndex) * inv_screen - 1.0f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;

    float2 scale = 1 / (float2(launchDim) * sqrt_num_samples) * 2.0f;

    for (int i = 0; i < sqrt_num_samples; ++i) {
        for (int j = 0; j < sqrt_num_samples; ++j) {

            float2 jitter = float2((i+1) + nextRand(seed), (j+1) + nextRand(seed));
            float2 d = pixel + jitter * jitter_scale;
            d.y = -d.y;
            float3 ray_origin = eye.xyz;
            float3 ray_direction = normalize(d.x * U.xyz * fov + d.y * V.xyz * fov + W.xyz);

            RayDesc ray;
            ray.Origin = ray_origin;
            ray.Direction = ray_direction;
            ray.TMin = 0.001;
            ray.TMax = 500;

            RayPayload payload;
            payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
            payload.shadow = float4(0.0f, 0.0f, 0.0f, 0.0f);
            payload.aoValue = 0.0f;
            payload.depth = 1;
            payload.seed = seed;

            TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
            result += payload.result;
            shadow += payload.shadow;
            aoValue += payload.aoValue;
        }
    }

    if (mask[launchIndex].r == 1.0f)
        output0[launchIndex] = result / samples;
    else
        output0[launchIndex] = float4(0, 0, 0, 0);
    output1[launchIndex] = aoValue / samples;
    output2[launchIndex] = shadow / samples;

#else

    uint seed = initRand(launchIndex.x + launchIndex.y * launchDim.x, frameCount);
    
    float2 crd = float2(launchIndex);
    float2 dims = float2(launchDim);

    float2 d = ((crd / dims) * 2.0f - 1.0f);
    d.y = -d.y;

    float3 u = float3(U.xyz);
    float3 v = float3(V.xyz);
    float3 w = float3(W.xyz);
    
    RayDesc ray;
    ray.Origin = eye.xyz;
    ray.Direction = normalize(d.x*u*fov + d.y*v*fov + w);
    ray.TMin = 0.001;
    ray.TMax = 500;

    RayPayload payload;
    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
    payload.shadow = float4(0.0f, 0.0f, 0.0f, 0.0f);
    payload.aoValue = 0.0f;
    payload.depth = 1;
    payload.seed = seed;
    TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);

    if (mask[launchIndex].r == 1.0f) {
        output0[launchIndex] = float4(payload.result.rgb, 1.0f);
    }
    else {
        output0[launchIndex] = float4(payload.result.rgb, 0.0f);
    }
    output1[launchIndex] = payload.aoValue;
    output2[launchIndex] = payload.shadow;
#endif
    
}


[shader("closesthit")]
void shade(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs) {

    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    RayPayload rp;
    rp.result = float4(1, 1, 1, 1);
    rp.shadow = float4(0, 0, 0, 0);

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

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            float3 newDir = sampleUnitHemisphereCosWeighted(n, payload.seed);
            RayDesc ray;
            ray.Origin = hit_point;
            ray.Direction = newDir;
            ray.TMin = 0.0001f;
            ray.TMax = 1000.0f;
            TraceRay(gRtScene, 0x04, 0xFF, 0, 2, 0, ray, rp);

            aoValues += rp.aoValue;
        }
    }

    float4 shadow = sampleAreaLight(n, hit_point, float3(lightPos.xyz), float3(0, -1, 0), lightSize, payload.seed);
    shadow = max(shadow, float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    MaterialAttrs mattr = materialAttributes[InstanceID()];
    float4 color = mattr.diffuse;

    if (mattr.texCount > 0) {
        float2 texCoord = float2(0, 0);
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

        color = color * tex[InstanceID()][uint2(texCoord * tsize)];
    }

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);
}


[shader("miss")]
void miss(inout RayPayload payload) {

    payload.result = float4(0.53f, 0.81f, 0.92f, 1.0f);
    payload.shadow = float4(1.0f, 1.0f, 1.0f, 1.0f);
    payload.aoValue = 1.0f;
}


[shader("anyhit")]
void anyHitShadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    payload.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
}


[shader("miss")]
void missShadow(inout RayPayload payload) {

    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("closesthit")]
void shadeGlass(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    float3 hit_point = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint tIndex = PrimitiveIndex();
    int address = tIndex * 3 * 4;
    uint3 indices = index[InstanceID()].Load3(address);

    float3 an = float3(0, 0, 0);
    for(uint i = 0; i < 3; ++i) {
        uint vaddr = indices[i] * 4 * 4;
        uint4 tan = normal[InstanceID()].Load4(vaddr);
        an += asfloat(tan.xyz) * barycentrics[i];
    }

    float3 n = normalize(an);

    float fresnel = dot(-normalize(ObjectRayDirection()), n);
    float ratio = F + (1.0f - F) * pow((1.0f - fresnel), FresnelPower);

    float3 reflectDir = reflect(ObjectRayDirection(), n);
    float3 refractDir = ObjectRayDirection();

    RayDesc ray;
    ray.Origin = hit_point;
    ray.TMin = 0.001f;
    ray.TMax = 5000;

    RayPayload refrPayload = payload;
    ray.Direction = refractDir;
    TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, refrPayload);

    RayPayload reflPayload = payload;
    ray.Direction = reflectDir;
    TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, reflPayload);

    payload.result = float4((lerp(refrPayload.result, reflPayload.result, ratio) * 0.9f).rgb, 1);
    payload.shadow = refrPayload.shadow * 0.9f;
}


[shader("anyhit")]
void glassShadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

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
    atenuation *= sqrt(abs(dot(n, WorldRayDirection())));
    
    payload.result *= atenuation;
    payload.shadow = atenuation;

    IgnoreHit();
}


[shader("anyhit")]
void anyHitGrade(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

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
void gradeShadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
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