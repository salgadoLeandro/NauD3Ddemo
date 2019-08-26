Texture2D texUnit : register(t0);
SamplerState samp : register(s0);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

cbuffer constantsFrag2 : register(b1) {
    float4 emission;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    
    float4 c = texUnit.Sample(samp, input.TexCoord);
    return emission * c;
}