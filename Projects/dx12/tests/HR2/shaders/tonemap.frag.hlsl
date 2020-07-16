struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texPos : TEXCOORD0;
};

Texture2D tex : register(t0);
SamplerState s0 : register(s0);

float4 main(VS_OUTPUT input) : SV_TARGET {

    float3 c = tex.Sample(s0, input.texPos).rgb;

    return float4(pow(c, 1.0f / 2.2f), 1.0f);
}