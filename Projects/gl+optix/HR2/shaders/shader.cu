#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_aabb_namespace.h>
#include "random.h"

using namespace optix;

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         fov, , );

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
rtBuffer<float, 2> output1;
rtBuffer<float4, 2> output2;

rtTextureSampler<float4, 2> mask;

struct PerRayDataResult {
    float4 result;
    float4 shadow;
    float aoValue;
    int depth;
    uint seed;
};

__device__ float FresnelPower = 0.9f;
__device__ float F = 0.039962f;

__device__ float2 lightSize = {0.4, 0.45};


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(PerRayDataResult, payload, rtPayload, );

rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(float3, texCoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(int, Phong, , );
rtDeclareVariable(int, Shadow, , );

#include "util.h"

RT_PROGRAM void raygen() {
    size_t2 screen = output0.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frameCount);
    
    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.0f - 1.0f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U*fov + d.y*V*fov + W);

    optix::Ray r = optix::make_Ray(ray_origin, ray_direction, Phong, 0.00000000001, RT_DEFAULT_MAX);

    PerRayDataResult payld;
    payld.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    payld.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    payld.aoValue = 0.0f;
    payld.depth = 1;
    payld.seed = seed;
    
    rtTrace(top_object, r, payld);

    float4 i = tex2D(mask, launch_index.x, launch_index.y);
    float3 clr = i.x == 1.0f ? make_float3(payld.result.x, payld.result.y, payld.result.z) : make_float3(1.0f, 1.0f, 1.0f);
    
    output0[launch_index] = make_float4(clr, 1.0f);
    output1[launch_index] = payld.aoValue;
    output2[launch_index] = payld.shadow;
}


RT_PROGRAM void exception() {
    output0[launch_index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}


RT_PROGRAM void shade() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 1;
    uint lightSamples = 1;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}


RT_PROGRAM void miss() {

    payload.result = make_float4(0.53f, 0.81f, 0.92f, 1.0f);
    payload.shadow = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    payload.aoValue = 1.0f;
}


RT_PROGRAM void anyHitShadow() {

    payload.result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}


RT_PROGRAM void missShadow() {

    payload.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}


RT_PROGRAM void shadeGlass() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

    float fresnel = dot(-normalize(ray.direction), n);
    float ratio = F + (1.0f - F) * pow((1.0f - fresnel), FresnelPower);

    float3 reflectDir = reflect(ray.direction, n);
    float3 refractDir = ray.direction;
    
    optix::Ray refl_ray(hit_point, reflectDir, Phong, 0.001, 5000);
    optix::Ray refr_ray(hit_point, refractDir, Phong, 0.001, 5000);

    PerRayDataResult reflPayload = payload;
    PerRayDataResult refrPayload = payload;

    rtTrace(top_object, refl_ray, reflPayload);
    rtTrace(top_object, refr_ray, refrPayload);

    float4 tmp = lerp(refrPayload.result, reflPayload.result, ratio) * 0.9f;
    payload.result = make_float4(tmp.x, tmp.y, tmp.z, 1.0f);
    payload.shadow = refrPayload.shadow * 0.9f;
}


RT_PROGRAM void glassShadow() {

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float atenuation = 1.0f;
    atenuation *= sqrt(abs(dot(n, ray.direction)));

    payload.result *= atenuation;
    payload.shadow *= atenuation;

    rtIgnoreIntersection();
}


RT_PROGRAM void anyHitGrade() {
    if (tex2D(tex0, texCoord.x, texCoord.y).w < 0.25f)
        rtIgnoreIntersection();
}


RT_PROGRAM void gradeShadow() {
    if (tex2D(tex0, texCoord.x, texCoord.y).w < 0.25f)
        rtIgnoreIntersection();
    else {
        payload.result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        rtTerminateRay();
    }
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

RT_PROGRAM void shade_1_4() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 1;
    uint lightSamples = 4;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_1_8() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 1;
    uint lightSamples = 8;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_1_16() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 1;
    uint lightSamples = 16;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_4_1() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 4;
    uint lightSamples = 1;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_4_4() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 4;
    uint lightSamples = 4;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_4_8() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 4;
    uint lightSamples = 8;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_4_16() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 4;
    uint lightSamples = 16;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_8_1() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 8;
    uint lightSamples = 1;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_8_4() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 8;
    uint lightSamples = 4;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_8_8() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 8;
    uint lightSamples = 8;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}

RT_PROGRAM void shade_8_16() {
    if (payload.depth >= 4) return;

    if (payload.depth < 0) {
        payload.aoValue = 0.0f;
        return;
    }

    uint aoSamples = 8;
    uint lightSamples = 16;

    PerRayDataResult rp;
    rp.result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    rp.shadow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 newDir;

    float aoValues = payload.aoValue;
    if (payload.depth <= 1) {
        rp.depth = -1;
        for(uint i = 0; i < aoSamples; ++i) {
            sampleUnitHemisphereCosWeighted(n, newDir, payload.seed);
            optix::Ray aoRay(hit_point, newDir, Phong, 0.2, 5000000);
            rtTrace(top_object, aoRay, rp);
            
            aoValues += rp.aoValue;
        }
    }

    float4 shadow = make_float4(0.0f);
    for(uint i = 0; i < lightSamples; ++i) {
        shadow += sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0, -1, 0), lightSize.x, lightSize.y, payload.seed);
    }
    shadow = fmaxf((shadow / lightSamples), make_float4(0.025, 0.025, 0.025, 1));
    payload.shadow = shadow;

    float4 color = diffuse;
    if (texCount > 0)
        color = color * tex2D(tex0, texCoord.x, texCoord.y);

    payload.result *= (color * shadow);
    payload.aoValue += (aoValues / aoSamples);

}