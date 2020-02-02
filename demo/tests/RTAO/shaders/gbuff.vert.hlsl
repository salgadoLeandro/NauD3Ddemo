struct VS_INPUT {
    float4 position : POSITION;
    float4 normal : NORMAL;
    float4 texCoord : TEXCOORD0;
};

cbuffer Constants : register(b0) {
	float4x4 m_pvm;
    float4x4 m_model;
    float4 cameraPos;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 worldPos : COLOR0;
    float4 worldNor : COLOR1;
};

VS_OUTPUT main ( VS_INPUT input ) {
    
    VS_OUTPUT output;
    output.position = mul(m_pvm, input.position);
    output.worldPos = mul(m_model, input.position);
    float3 norm = mul(m_model, input.normal).xyz;
    float w = length(output.worldPos - cameraPos);
    output.worldNor = float4(norm, w);
    return output;
}