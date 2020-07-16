Texture2D t1 : register(t0);
SamplerState s1 : register(s0);
Texture2D t2 : register(t1);
SamplerState s2 : register(s1);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 texCoord : TEXCOORD0;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    
    float4 cr = t1.Sample(s1, input.texCoord.xy);
    float4 rr = t2.Sample(s2, input.texCoord.xy);

    return cr.w == 0.5f ? rr : cr;
}