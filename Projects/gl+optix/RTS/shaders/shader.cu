#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_aabb_namespace.h>
#include "random.h"

using namespace optix;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float, fov, , );

rtDeclareVariable(float4, diffuse, , );
rtDeclareVariable(int, texCount, , );

rtDeclareVariable(float4, lightDir, , );
rtDeclareVariable(float4, lightPos, , );
rtDeclareVariable(uint, frameCount, , );

rtDeclareVariable(rtObject, top_object, , );

rtBuffer<float4> vertex_buffer;
rtBuffer<uint> index_buffer;
rtBuffer<float4> normal;
rtBuffer<float4> texCoord0;
rtTextureSampler<float4, 2> tex0;

rtBuffer<float4, 2> output0;

struct PerRayDataResult {
    float4 result;
    int depth;
    uint seed;
    float entrance;
};

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(PerRayDataResult, payload, rtPayload, );

rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float3, texCoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(int, Phong, , );
rtDeclareVariable(int, Shadow, , );

__device__ float FresnelPower = 0.9f;
__device__ float F = 0.039962f;

__device__ float2 lightSize = {0.4, 0.45};

#include "util.h"

RT_PROGRAM void raygen() {
    size_t2 screen = output0.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frameCount);

    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.0f - 1.0f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U*fov + d.y*V*fov + W);

    optix::Ray r = optix::make_Ray(ray_origin, ray_direction, Phong, 0.00000000001, RT_DEFAULT_MAX);

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.depth = 1;
    rp.seed = seed;
    rp.entrance = 0.0f;

    rtTrace(top_object, r, rp);

    output0[launch_index] = rp.result;
}


RT_PROGRAM void exception() {
    output0[launch_index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}


RT_PROGRAM void tracePath() {
    if (payload.depth >= 4) return;
    
    uint lightSamples = 1;
    
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    
    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos.x, lightPos.y, lightPos.z), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow.w = lightSamples;
    shadow = shadow / make_float4(lightSamples);
    
    payload.result *= shadow;
}


RT_PROGRAM void miss() {
    payload.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}


RT_PROGRAM void missShadow() {
    payload.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}


RT_PROGRAM void any_hit_shadow() {
    payload.result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}


RT_PROGRAM void alpha_test() {
    if (tex2D(tex0, texCoord.x, texCoord.y).w < 0.25f)
        rtIgnoreIntersection();
}


RT_PROGRAM void alpha_test_shadow() {
    if (tex2D(tex0, texCoord.x, texCoord.y).w < 0.25f)
        rtIgnoreIntersection();
    else {
        payload.result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        rtTerminateRay();
    }
}


RT_PROGRAM void traceGlass() {
    if (payload.depth >= 4) return;

    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 dir = ray.direction;

    optix::Ray nray(hit_point, dir, Phong, 0.00001, 500000);

    PerRayDataResult rpd = payload;

    rtTrace(top_object, nray, rpd);

    payload.result = rpd.result * 0.9f;
}


RT_PROGRAM void keepGoingShadow() {
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float atenuation = 1.0f;
    atenuation *= sqrt(abs(dot(n, ray.direction)));

    payload.result *= atenuation;

    rtIgnoreIntersection();
}


RT_PROGRAM void geometryintersection(int primIdx) {

	float4 vecauxa = vertex_buffer[index_buffer[primIdx*3]];
	float4 vecauxb = vertex_buffer[index_buffer[primIdx*3+1]];
	float4 vecauxc = vertex_buffer[index_buffer[primIdx*3+2]];

	float3 v0 = make_float3(vecauxa);
	float3 v1 = make_float3(vecauxb);
	float3 v2 = make_float3(vecauxc);

    float3 n;
    float  t, beta, gamma;
    if( intersect_triangle( ray, v0, v1, v2, n, t, beta, gamma ) ) {

        if(  rtPotentialIntersection( t ) ) {

            float3 n0 = make_float3(normal[ index_buffer[primIdx*3]]);
            float3 n1 = make_float3(normal[ index_buffer[primIdx*3+1]]);
            float3 n2 = make_float3(normal[ index_buffer[primIdx*3+2]]);

            float3 t0 = make_float3(texCoord0[ index_buffer[primIdx*3]]);
            float3 t1 = make_float3(texCoord0[ index_buffer[primIdx*3+1]]);
            float3 t2 = make_float3(texCoord0[ index_buffer[primIdx*3+2]]);

            shading_normal   = normalize( n0*(1.0f-beta-gamma) + n1*beta + n2*gamma );
            texCoord =  t0*(1.0f-beta-gamma) + t1*beta + t2*gamma ;
            geometric_normal = normalize( n );

            rtReportIntersection(0);
        }
    }
}


RT_PROGRAM void boundingbox(int primIdx, float result[6]) {

	float3 v0 = make_float3(vertex_buffer[index_buffer[primIdx*3]]);
	float3 v1 = make_float3(vertex_buffer[index_buffer[primIdx*3+1]]);
	float3 v2 = make_float3(vertex_buffer[index_buffer[primIdx*3+2]]);  
	
	const float  area = length(cross(v1-v0, v2-v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if(area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf( fminf( v0, v1), v2 );
		aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
	} 
	else {
	    aabb->invalidate();
	}
}

RT_PROGRAM void tracePath_4() {
    if (payload.depth >= 4) return;
    
    uint lightSamples = 4;
    
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    
    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos.x, lightPos.y, lightPos.z), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow.w = lightSamples;
    shadow = shadow / make_float4(lightSamples);
    
    payload.result *= shadow;
}

RT_PROGRAM void tracePath_8() {
    if (payload.depth >= 4) return;
    
    uint lightSamples = 8;
    
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    
    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos.x, lightPos.y, lightPos.z), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow.w = lightSamples;
    shadow = shadow / make_float4(lightSamples);
    
    payload.result *= shadow;
}

RT_PROGRAM void tracePath_16() {
    if (payload.depth >= 4) return;
    
    uint lightSamples = 16;
    
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    
    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos.x, lightPos.y, lightPos.z), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow.w = lightSamples;
    shadow = shadow / make_float4(lightSamples);
    
    payload.result *= shadow;
}

RT_PROGRAM void tracePath_32() {
    if (payload.depth >= 4) return;
    
    uint lightSamples = 32;
    
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    
    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos.x, lightPos.y, lightPos.z), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow.w = lightSamples;
    shadow = shadow / make_float4(lightSamples);
    
    payload.result *= shadow;
}