struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 color : COLOR;
    float4 texCoord : TEXCOORD0;
};

struct OutputRT {
    float4 rt1 : SV_TARGET0;
    float4 rt2 : SV_TARGET1;
};

OutputRT main ( VS_OUTPUT input ) {
    OutputRT outp;
    outp.rt1 = input.texCoord;
    outp.rt2 = float4(input.color.xyz, 1.0f);
    return outp;
}