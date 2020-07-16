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

rtTextureSampler<float4, 2> gPos;
rtTextureSampler<float4, 2> gNorm;

struct PerRayDataResult {
	float4 result;
    float aoValue;
};

__device__ float gAORadius = 1000.0f;
__device__ float gMinT = 0.0001f;

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

#include "util.h"

__device__ float3 getPerpendicularVector(float3 u) {
	float3 a = make_float3(fabsf(u.x), fabsf(u.y), fabsf(u.z));
	uint xm = ((a.x - a.y) < 0 && (a.x - a.z) < 0) ? 1 : 0;
	uint ym = (a.y - a.z) < 0 ? (1 ^ xm) : 0;
	uint zm = 1 ^ (xm | ym);
	return cross(u, make_float3(xm, ym, zm));
}

__device__ float3 getCosHemisphereSample(uint& seed, float3 hitNorm) {

	float2 randVal = make_float2(rnd(seed), rnd(seed));

	float3 bitangent = getPerpendicularVector(hitNorm);
	float3 tangent = cross(bitangent, hitNorm);
	float r = sqrt(randVal.x);
	float phi = 2.0f * 3.14159265f * randVal.y;

	return tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + hitNorm * sqrt(1 - randVal.x);
}

__device__ float shootAOray(float3 orig, float3 dir, float minT, float maxT) {

	PerRayDataResult rp;
	rp.aoValue = 0.0f;

	optix::Ray aoRay = optix::make_Ray(orig, dir, Phong, minT, maxT);
	rtTrace(top_object, aoRay, rp);

	return rp.aoValue;
}


RT_PROGRAM void raygen() {
	uint gNumRays = 4;

	size_t2 screen = output0.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frameCount);

	float4 worldPos = tex2D(gPos, launch_index.x, launch_index.y);
	float4 worldNor = tex2D(gNorm, launch_index.x, launch_index.y);

	float ambientOcclusion = 0.0f;
	
	float3 wn = make_float3(worldNor.x, worldNor.y, worldNor.z);
	float3 wp = make_float3(worldPos.x, worldPos.y, worldPos.z);

	for(uint i = 0; i < gNumRays; ++i) {
		float3 worldDir = getCosHemisphereSample(seed, wn);

		ambientOcclusion += shootAOray(wp, worldDir, gMinT, gAORadius);
	}

	float aoColor = ambientOcclusion / __int2float_rn(gNumRays);
	output0[launch_index] = make_float4(aoColor, aoColor, aoColor, 1.0f);
}


RT_PROGRAM void exception() {
	output0[launch_index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}


RT_PROGRAM void aoAnyHit() {
	if (tex2D(tex0, texCoord.x, texCoord.y).w < 0.25f)
		rtIgnoreIntersection();
}


RT_PROGRAM void aoMiss() {
	payload.aoValue = 1.0f;
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

RT_PROGRAM void raygen_8() {
	uint gNumRays = 8;

	size_t2 screen = output0.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frameCount);

	float4 worldPos = tex2D(gPos, launch_index.x, launch_index.y);
	float4 worldNor = tex2D(gNorm, launch_index.x, launch_index.y);

	float ambientOcclusion = 0.0f;
	
	float3 wn = make_float3(worldNor.x, worldNor.y, worldNor.z);
	float3 wp = make_float3(worldPos.x, worldPos.y, worldPos.z);

	for(uint i = 0; i < gNumRays; ++i) {
		float3 worldDir = getCosHemisphereSample(seed, wn);

		ambientOcclusion += shootAOray(wp, worldDir, gMinT, gAORadius);
	}

	float aoColor = ambientOcclusion / __int2float_rn(gNumRays);
	output0[launch_index] = make_float4(aoColor, aoColor, aoColor, 1.0f);
}

RT_PROGRAM void raygen_16() {
	uint gNumRays = 16;

	size_t2 screen = output0.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frameCount);

	float4 worldPos = tex2D(gPos, launch_index.x, launch_index.y);
	float4 worldNor = tex2D(gNorm, launch_index.x, launch_index.y);

	float ambientOcclusion = 0.0f;
	
	float3 wn = make_float3(worldNor.x, worldNor.y, worldNor.z);
	float3 wp = make_float3(worldPos.x, worldPos.y, worldPos.z);

	for(uint i = 0; i < gNumRays; ++i) {
		float3 worldDir = getCosHemisphereSample(seed, wn);

		ambientOcclusion += shootAOray(wp, worldDir, gMinT, gAORadius);
	}

	float aoColor = ambientOcclusion / __int2float_rn(gNumRays);
	output0[launch_index] = make_float4(aoColor, aoColor, aoColor, 1.0f);
}

RT_PROGRAM void raygen_32() {
	uint gNumRays = 32;

	size_t2 screen = output0.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frameCount);

	float4 worldPos = tex2D(gPos, launch_index.x, launch_index.y);
	float4 worldNor = tex2D(gNorm, launch_index.x, launch_index.y);

	float ambientOcclusion = 0.0f;
	
	float3 wn = make_float3(worldNor.x, worldNor.y, worldNor.z);
	float3 wp = make_float3(worldPos.x, worldPos.y, worldPos.z);

	for(uint i = 0; i < gNumRays; ++i) {
		float3 worldDir = getCosHemisphereSample(seed, wn);

		ambientOcclusion += shootAOray(wp, worldDir, gMinT, gAORadius);
	}

	float aoColor = ambientOcclusion / __int2float_rn(gNumRays);
	output0[launch_index] = make_float4(aoColor, aoColor, aoColor, 1.0f);
}