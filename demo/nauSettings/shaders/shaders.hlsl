RaytracingAccelerationStructure gRtScene : register(t0);
RWTexture2D<float4> gOutput : register(u0);
RWTexture2D<float4> posBuffer : register(u1);
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

struct RayPayload {

	float3 color;
};

struct ShadowPayload {

	bool hit;
};


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
		//ray.Origin = float3(0, 0, -2);
		//ray.Direction = normalize(d.x*u*fov + d.y*v*fov + w);
		ray.Direction = normalize(float3(d.x * aspectRatio, d.y, 1));

		ray.TMin = 0;
		ray.TMax = 100000;

		RayPayload payload;
		TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
		float3 col = payload.color;
		gOutput[launchIndex.xy] = float4(col, 1.0);
	}
	else {
		gOutput[launchIndex.xy] = float4(1.0f, 1.0f, 1.0f, 0.0f);
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
	//ray.Origin = float3(0, 0, -2);
	//ray.Direction = normalize(d.x*u*fov + d.y*v*fov + w);
	ray.Direction = normalize(float3(d.x * aspectRatio, d.y, 1));

	ray.TMin = 0;
	ray.TMax = 100000;

	RayPayload payload;
	TraceRay(gRtScene, 0, 0xFF, 0, 2, 0, ray, payload);
	float3 col = payload.color;
	gOutput[launchIndex.xy] = float4(col, 1.0);
}


[shader("miss")]
void miss(inout RayPayload payload) {

	payload.color = float3(1.0, 1.0, 1.0);
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

	payload.color = float3(1.0, 0.0, 0.0);
}


[shader("closesthit")]
void chit3(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	payload.color = float3(0.5, 0.5, 0.5);
}


[shader("anyhit")]
void shadowahit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	payload.hit = true;
}

[shader("anyhit")]
void shadowahit2(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

	payload.hit = true;
}


[shader("miss")]
void shadowMiss(inout ShadowPayload payload) {

	payload.hit = false;
}