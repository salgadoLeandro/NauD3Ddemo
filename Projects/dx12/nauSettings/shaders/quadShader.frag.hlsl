Texture2D texUnit : register(t0);
SamplerState samp : register(s0);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texCoordV : TEXCOORD;
};

float4 main(VS_OUTPUT input) : SV_TARGET {
	return texUnit.Sample(samp, input.texCoordV);
}