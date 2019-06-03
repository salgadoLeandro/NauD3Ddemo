Texture2D t1 : register(t0);
SamplerState s1 : register(s0);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

cbuffer Constantsf : register(b1) {
    float4 emission;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    
    float4 c = t1.Sample(s1, input.TexCoord);
    return emission * c;
}