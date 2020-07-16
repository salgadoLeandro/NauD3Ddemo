RaytracingAccelerationStructure gRtScene : register(t0);

RWTexture2D<float4> output0 : register(u0);
RWTexture2D<float4> posBuffer : register(u1);

ByteAddressBuffer position[5] : register(t1);
ByteAddressBuffer texCoord0[5] : register(t6);
ByteAddressBuffer index[5] : register(t11);

Texture2D<float4> tex[5] : register(t16);

cbuffer Camera : register(b0) {
	float4 eye;
	float4 V;
	float4 U;
	float4 W;
	float fov;
};

cbuffer GlobalAttributes : register(b1) {
	float4 lightDir;
};

struct MaterialAttrs {
	float4 diffuse;
	int texCount;
};

cbuffer MaterialAttributes : register(b2) {
	MaterialAttrs materialAttributes[5];
};

struct RayPayload {

	float3 color;
};

struct ShadowPayload {

	bool hit;
};

static const float PI = 3.14159265f;

[shader("raygeneration")]
void rayGen() {  

    uint3 launchIndex = DispatchRaysIndex();
	uint3 launchDim = DispatchRaysDimensions();

	float4 r = posBuffer[launchIndex.xy];

	if (r.w == 0.5f) {
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
		TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
		float3 col = payload.color;
		output0[launchIndex.xy] = float4(col, 1.0);
	}
	else {
		output0[launchIndex.xy] = float4(1.0f, 1.0f, 1.0f, 0.0f);
	}
}


[shader("raygeneration")]
void rayGen2() {  

    uint3 launchIndex = DispatchRaysIndex();
	uint3 launchDim = DispatchRaysDimensions();

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
	TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
	float3 col = payload.color;
	output0[launchIndex.xy] = float4(col, 1.0);
}


[shader("miss")]
void miss(inout RayPayload payload) {

	payload.color = float3(1.0, 1.0, 1.0);
}

[shader("closesthit")]
void closesthit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	float hitT = RayTCurrent();
	float3 rayDirW = WorldRayDirection();
	float3 rayOriginW = WorldRayOrigin();

	float3 posW = rayOriginW + hitT * rayDirW;

	RayDesc ray;
	ray.Origin = posW;
	ray.Direction = normalize(float3(-lightDir.xyz));
	ray.TMin = 0.01;
	ray.TMax = 100000;

	ShadowPayload shadowPayload;

	TraceRay(gRtScene, 0, 0xFF, 1, 0, 1, ray, shadowPayload);

	float shadowfactor = shadowPayload.hit ? 0.25 : 1.0;

	uint tIndex = PrimitiveIndex();
	int address = tIndex * 3 * 4;
	uint3 indices = index[InstanceID()].Load3(address);

	float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

	MaterialAttrs mattr = materialAttributes[InstanceID()];

	if (mattr.texCount > 0) {
		float2 texCoord = float2(0.0, 0.0);
		for(uint i = 0; i < 3; ++i){
			int vaddr = indices[i] * 4 * 4;
			uint4 texc = texCoord0[InstanceID()].Load4(vaddr);
			texCoord += asfloat(texc.xy) * barycentrics[i];
		}
		uint2 size;
		tex[InstanceID()].GetDimensions(size.x, size.y);

		payload.color = tex[InstanceID()][uint2(texCoord * size)] * shadowfactor;
	}
	else {
		payload.color = shadowfactor;
	}

	
}


[shader("closesthit")]
void chit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

	payload.color = barycentrics;

	float hitT = RayTCurrent();
	float3 rayDirW = WorldRayDirection();
	float3 rayOriginW = WorldRayOrigin();

	float3 posW = rayOriginW + hitT * rayDirW;
	RayDesc ray;
	ray.Origin = posW;
	ray.Direction = normalize(float3(-lightDir.xyz));
	ray.TMin = 0.01;
	ray.TMax = 100000;
	
	ShadowPayload shadowPayload;
	
	TraceRay(gRtScene, 0, 0xFF, 1, 0, 1, ray, shadowPayload);

	float factor = shadowPayload.hit ? 0.1 : 1.0;
	payload.color = payload.color * factor;
}


[shader("closesthit")]
void chit2(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

	float hitT = RayTCurrent();
	float3 rayDirW = WorldRayDirection();
	float3 rayOriginW = WorldRayOrigin();

	float3 posW = rayOriginW + hitT * rayDirW;

	RayDesc ray;
	ray.Origin = posW;
	ray.Direction = normalize(float3(-lightDir.xyz));
	ray.TMin = 0.01;
	ray.TMax = 100000;
	
	ShadowPayload shadowPayload;
	
	TraceRay(gRtScene, 0, 0xFF, 1, 0, 1, ray, shadowPayload);

	float factor = shadowPayload.hit ? 0.25 : 1.0;
	payload.color = float3(1.0, 0.0, 0.0) * factor;

	uint tIndex = PrimitiveIndex();
	int address = tIndex * 3 * 4;
	uint3 indices = index[InstanceID()].Load3(address);

	float2 texCoord = float2(0.0, 0.0);
	for(uint i = 0; i < 3; ++i){
		int vaddr = indices[i] * 4 * 4;
		uint4 texc = texCoord0[InstanceID()].Load4(vaddr);
		texCoord += asfloat(texc.xy) * barycentrics[i];
	}

	uint2 size;
	tex[InstanceID()].GetDimensions(size.x, size.y);

	MaterialAttrs mattr = materialAttributes[InstanceID()];

	payload.color = tex[InstanceID()][uint2(texCoord * size)] * mattr.texCount;
}


[shader("closesthit")]
void chit3(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	float hitT = RayTCurrent();
	float3 rayDirW = WorldRayDirection();
	float3 rayOriginW = WorldRayOrigin();

	float3 posW = rayOriginW + hitT * rayDirW;

	RayDesc ray;
	ray.Origin = posW;
	ray.Direction = normalize(float3(-lightDir.xyz));
	ray.TMin = 0.01;
	ray.TMax = 100000;
	
	ShadowPayload shadowPayload;
	
	TraceRay(gRtScene, 0, 0xFF, 1, 0, 1, ray, shadowPayload);

	float factor = shadowPayload.hit ? 0.1 : 1.0;
	payload.color = float3(0.5, 0.5, 0.5) * factor;
}


[shader("anyhit")]
void shadowahit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	payload.hit = true;
}

[shader("miss")]
void shadowMiss(inout ShadowPayload payload) {

	payload.hit = false;
}