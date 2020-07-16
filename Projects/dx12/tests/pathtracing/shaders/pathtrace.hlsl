
#define mod(x, y) (x - (y * int(x / y)))

RaytracingAccelerationStructure gRtScene : register(t0);

RWTexture2D<float4> output0 : register(u0);

ByteAddressBuffer position[25] : register(t100);
ByteAddressBuffer normal[25] : register(t200);
ByteAddressBuffer texCoord0[25] : register(t300);
ByteAddressBuffer index[25] : register(t400);

Texture2D<float4> tex[25] : register(t500);

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
    float exposure;
    uint frameCount;
    int trace;
};

struct MaterialAttrs {
    float4 diffuse;
    int texCount;
};

cbuffer MaterialAttributes : register(b2) {
    MaterialAttrs materialAttributes[25];
};

struct RayPayload {
    float4 result;
    int depth;
    uint seed;
    float entrance;
};


static const float M_PI = 3.14159265f;

#define mod(x, y) (x - (y * int(x / y)))

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
    shadow_prd.result = float4(0, 0, 0, 0);
    float4 result = float4(0, 0, 0, 0);
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
        if(NdotL > 0) {
            dot1 = max(0.0f, dot(lightN, -lDir));

            shadow_prd.result = float4(1, 1, 1, 1);

            RayDesc ray;
            ray.Origin = hitPoint;
            ray.Direction = lDir;
            ray.TMin = 0.001f;
            ray.TMax = lightDist + 0.01f;

            TraceRay(gRtScene, 0x04, 0xFF, 1, 0, 1, ray, shadow_prd);
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


[shader("raygeneration")]
void pinhole_camera_ms() {

    float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    int sqrt_num_samples = 2;
    int samples = sqrt_num_samples * sqrt_num_samples;
    uint seedi, seedj;

    uint3 launchIndex = DispatchRaysIndex();
    uint3 launchDim = DispatchRaysDimensions();

    uint2 screen;
    output0.GetDimensions(screen.x, screen.y);

    float2 inv_screen = 1.0f / float2(screen) * 2.0f;
    float2 pixel = (float2(launchIndex.xy)) * inv_screen - 1.0f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;

    float2 scale = 1 / (float2(launchDim.xy) * sqrt_num_samples) * 2.0f;
    uint seed = initRand(screen.x * launchIndex.x + launchIndex.y, frameCount);

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
            ray.TMin = 0.000000001;
            ray.TMax = 1000000;

            RayPayload payload;
            payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
            payload.depth = 1;
            payload.seed = seed;

            TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
            color += payload.result;
        }
    }

    output0[launchIndex.xy] = color / samples;
}


[shader("miss")]
void miss(inout RayPayload payload) {

    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("closesthit")]
void tracePath(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
    //if (payload.depth >= 2) return;

    RayPayload rp;
    rp.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 newDir;

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint tIndex = PrimitiveIndex();
    int address = tIndex * 3 * 4;
    uint3 indices = index[InstanceID()].Load3(address);

    /*float3 an = float3(0.0, 0.0, 0.0);
    float2 texCoord = float2(0.0f, 0.0f);
    for(uint i = 0; i < 3; ++i) {
        uint vaddr = indices[i] * 4 * 4;
        uint4 tan = normal[InstanceID()].Load4(vaddr);
        uint4 texc = texCoord0[InstanceID()].Load4(vaddr);
        an += asfloat(tan.xyz) * barycentrics[i];
        texCoord += asfloat(texc.xy) * barycentrics[i];
    }*/
    uint3 vaddr = indices * 4 * 4;
    ByteAddressBuffer cur_tex = texCoord0[InstanceID()];
    float2 texCoord = asfloat(cur_tex.Load4(vaddr.x).xy) * barycentrics.x
                    + asfloat(cur_tex.Load4(vaddr.y).xy) * barycentrics.y
                    + asfloat(cur_tex.Load4(vaddr.z).xy) * barycentrics.z;
    ByteAddressBuffer cur_norm = normal[InstanceID()];
    float3 an = asfloat(cur_norm.Load4(vaddr.x).xyz) * barycentrics.x
              + asfloat(cur_norm.Load4(vaddr.y).xyz) * barycentrics.y
              + asfloat(cur_norm.Load4(vaddr.z).xyz) * barycentrics.z;


    float3 n = normalize(an);
    float3 hit_point = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    float4 shadow = sampleAreaLight(n, hit_point, float3(lightPos.xyz), float3(0, -1, 0), 0.4, 0.45, payload.seed);

    MaterialAttrs mattr = materialAttributes[InstanceID()];
    float4 color = mattr.diffuse;

    float p = max(color.x, max(color.y, color.z));
    color = color * 1.0 / (1 - p);
    if (payload.depth > 1 || nextRand(payload.seed) > p) {
        color = color * 1.0 / (1 - p);
    }
    else {
        rp.depth = payload.depth + 1;
        rp.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
        rp.seed = payload.seed;
        
       float3 newDir = sampleUnitHemisphereCosWeighted(n, payload.seed);
        
        RayDesc ray;
        ray.Origin = hit_point;
        ray.Direction = newDir;
        ray.TMin = 0.2;
        ray.TMax = 500000;
        TraceRay(gRtScene, 0x04, 0xFF, 0, 2, 0, ray, rp);

        shadow += rp.result;
    }

    if (mattr.texCount > 0) {
        /*float2 texCoord = float2(0.0f, 0.0f);
        for (uint i = 0; i < 3; ++i) {
            uint vaddr = indices[i] * 4 * 4;
            uint4 texc = texCoord0[InstanceID()].Load4(vaddr);
            texCoord += asfloat(texc.xy) * barycentrics[i];
        }*/
        uint2 tsize;
        tex[InstanceID()].GetDimensions(tsize.x, tsize.y);

        texCoord = float2(mod(texCoord.x, 1.0f), mod(texCoord.y, 1.0f));
        if (texCoord.x < 0) texCoord.x = 1 + texCoord.x;
        if (texCoord.y < 0) texCoord.y = 1 + texCoord.y;

        color = color * tex[InstanceID()][uint2(texCoord * tsize)];
    }

    payload.result *= color * shadow;
}


[shader("closesthit")]
void shadeLight(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
    //payload.result = float4(235.0f / 255.0f, 198.0f / 255.0f, 52.0f / 255.0f, 1.0f);
}


[shader("miss")]
void missShadow(inout RayPayload payload) {

    payload.result = float4(1.0f, 1.0f, 1.0f, 1.0f);
}


[shader("anyhit")]
void any_hit_shadow(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    
    payload.result = float4(0.0f, 0.0f, 0.0f, 0.0f);
}