Texture2D t1 : register(t0);
SamplerState s1 : register(s0);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 color : COLOR;
    float4 texCoord : TEXCOORD0;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    //return float4(1.0f, 0.0f, 0.0f, 1.0f);
    //return input.color;
    return t1.Sample(s1, input.texCoord.xy);
    //return input.texCoord;
}