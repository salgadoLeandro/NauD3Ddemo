RWTexture2D<float4> teste : register(u1);

#define mod(x, y) (x - (y * uint(x / y)))

[numthreads(1, 1, 1)]
void main(uint3 gID : SV_GroupID, uint3 dtID : SV_DispatchThreadID) {

	uint i = dtID.x * 256 + dtID.y;
	uint mi = mod(i, 256);

	float f = mi / 256.0f;
    teste[uint2(dtID.xy)] = float4(f, f, f, 1.0f);
}