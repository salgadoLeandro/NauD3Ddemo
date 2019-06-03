struct VS_INPUT {
    float4 position : POSITION;
    float4 normal : NORMAL;
    float4 texCoord : TEXCOORD0;
};

cbuffer PerVertexData : register(b0) {
	float4x4 m_pvm;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 color : COLOR;
    float4 texCoord : TEXCOORD0;
};

VS_OUTPUT main ( VS_INPUT input ) {
    VS_OUTPUT output;
    output.position = mul(m_pvm, input.position);
    output.color = abs(input.normal);
    output.texCoord = input.texCoord;
    return output;
}