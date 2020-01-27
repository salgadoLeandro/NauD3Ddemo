RWTexture2D<float> rtao : register(u0);
RWTexture2D<float4> outp : register(u1);

[numthreads(2, 2, 1)]
void main(uint2 dtID : SV_DispatchThreadID) {

    float4 c1 = outp[dtID];
    float t = rtao[dtID] * 0.1;
    float4 c2 = float4(t, t, t, 1);
    float4 cFinal = c1 * 0.9 + c2;

    outp[dtID] = cFinal;
}