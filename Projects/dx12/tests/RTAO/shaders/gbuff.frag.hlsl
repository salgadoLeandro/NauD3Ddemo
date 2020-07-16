struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 worldPos : COLOR0;
    float4 worldNor : COLOR1;
};

struct OutputRT {
    float4 rt1 : SV_TARGET0; //world position
    float4 rt2 : SV_TARGET1; //world normal
};

OutputRT main ( VS_OUTPUT input ) {

    OutputRT outp;
    outp.rt1 = input.worldPos;
    outp.rt2 = input.worldNor;
    return outp;
}