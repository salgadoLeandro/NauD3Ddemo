Texture2D t1 : register(t0);
SamplerState s1 : register(s0);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

cbuffer constantsFrag1 : register(b1) {
    float4 emission;
    int texCount;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    float4 outColor = emission;
    if (texCount != 0)
        outColor *= mul(t1.Sample(s1, input.TexCoord), emission);
    return outColor;
}