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

struct salReturn {
    float4 result;
    uint seed;
};


static const float M_PI = 3.14159265f;

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
            float3 lightN, float lightSizeX, float lightSizeY, inout uint seed) {
    
    RayPayload shadow_prd;
    shadow_prd.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 result = float4(0.0, 0.0f, 0.0f, 0.0f);
    uint lightSamples = 1;

    float2 r;
    float dot1 = dot(hitPoint - lightP, lightN);
    if (dot1 < 0) {
        return result;
    }
    
    for(uint i = 0; i < lightSamples; ++i) {
        
        r.x = nextRand(seed);
        r.y = nextRand(seed);

        float3 lPos = lightP + float3(1, 0, 0) * lightSizeX * r.x + float3(0, 0, 1) * lightSizeY * r.y;
        float3 lDir = lPos - hitPoint;
        float lightDist = length(lDir);
        lDir = normalize(lDir);

        float NdotL = dot(surfaceNormal, lDir);
        if (NdotL > 0) {
            dot1 = max(0.0f, dot(lightN, -lDir));

            shadow_prd.result = float4(1.0f, 1.0f, 1.0f, 1.0f);

            RayDesc ray;
            ray.Origin = hitPoint;
            ray.Direction = lDir;
            ray.TMin = 0.001;
            ray.TMax = lightDist + 0.01;

            TraceRay(gRtScene, 0, 0xFF, 1, 0, 1, ray, shadow_prd);
            result += shadow_prd.result * NdotL * dot1;
        }
    }

    result.w = lightSamples;
    
    return (result / lightSamples);
}


#define MS 1

[shader("raygeneration")]
void raygen() {
    uint2 launchIndex = DispatchRaysIndex();
	uint2 launchDim = DispatchRaysDimensions();

    uint2 screen;
    output0.GetDimensions(screen.x, screen.y);

    uint seed = initRand(screen.x * launchIndex.x + launchIndex.y, frameCount);

#if MS == 0

    float4 result = float4(0, 0, 0, 0);
    int sqrt_num_samples = 2;
    int samples = sqrt_num_samples * sqrt_num_samples;

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
            payload.depth = 1;
            payload.seed = seed;

            TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
            result += payload.result;
        }
    }

    output0[launchIndex] = result / samples;

#else
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
    payload.depth = 1;
    payload.seed = seed;
    payload.entrance = 0.0f;
	TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);

	output0[launchIndex.xy] = payload.result;
#endif
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

    float4 shadow = sampleAreaLight(n, hit_point, float3(lightPos.xyz), float3(0, -1, 0), 0.4, 0.45, payload.seed);

    MaterialAttrs mattr = materialAttributes[InstanceID()];
    float4 color = mattr.diffuse;

    /*if (mattr.texCount > 0) {
        float2 texCoord = float2(0.0f, 0.0f);
        for (uint i = 0; i < 3; ++i) {
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
    }*/

    payload.result *= color * shadow;
}

[shader("miss")]
void missShadow(inout RayPayload payload) {

    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("anyhit")]
void any_hit_shadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
    payload.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

static const float AtG = 0.6668f;
static const float FresnelPower = 0.9f;
static const float F = ((1.0-AtG) * (1.0-AtG)) / ((1.0+AtG) * (1.0+AtG));


[shader("closesthit")]
void traceGlass(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    if (payload.depth >= 4) return;

    float3 hit_point = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

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

    float fresnel = dot(-normalize(ObjectRayDirection()), n);
    float ratio = F + (1.0f - F) * pow((1.0 - fresnel), FresnelPower);

    float3 reflectDir = reflect(ObjectRayDirection(), n);
    //float3 refractDir = refract(ObjectRayDirection(), n, AtG);
    float3 refractDir = ObjectRayDirection();

    RayDesc ray;
    ray.Origin = hit_point;
    ray.TMin = 0.001;
    ray.TMax = 5000;

    RayPayload refrPayload = payload;
    
    ray.Direction = refractDir;
    TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, refrPayload);

    RayPayload reflPayload = payload;

    ray.Direction = reflectDir;
    TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, reflPayload);

    payload.result = (lerp(refrPayload.result, reflPayload.result, ratio) * 0.9f);
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