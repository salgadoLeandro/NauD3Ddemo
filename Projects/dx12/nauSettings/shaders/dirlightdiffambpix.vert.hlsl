struct VS_INPUT {
    float4 position : POSITION;
    float4 texCoord0 : TEXCOORD;
    float4 normal : NORMAL;
};

struct VS_OUTPUT {
    float4 vertexPos : SV_POSITION;
    float3 Normal : NORMAL;
    float3 LightDirection : COLOR;
    float2 TexCoord : TEXCOORD;
};

cbuffer constantsVert0 : register(b0) {
    float4x4 PVM;
    float4x4 V;
    float4 lightDirection;
    float3x3 NormalMatrix;
};

VS_OUTPUT main ( VS_INPUT input ) {
    VS_OUTPUT output;
    output.Normal = normalize(mul(NormalMatrix, float3(input.normal.xyz)));
    output.LightDirection = normalize(float3(mul(V, lightDirection).xyz));
    output.TexCoord = float2(input.texCoord0.xy);
    output.vertexPos = mul(PVM, input.position);
    return output;
}