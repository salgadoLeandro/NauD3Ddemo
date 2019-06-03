struct VS_INPUT {
    float4 position : POSITION;
    float4 texCoord0 : TEXCOORD;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

cbuffer ConstantBuffere : register(b0) {
    float4x4 PVM;
};

VS_OUTPUT main ( VS_INPUT input ) {
    VS_OUTPUT output;

    output.TexCoord = float2(input.texCoord0.xy);
    output.position = mul(PVM, input.position);

    return output;
}