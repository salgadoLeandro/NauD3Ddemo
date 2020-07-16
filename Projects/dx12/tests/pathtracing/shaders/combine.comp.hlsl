RWTexture2D<float4> tex1 : register(u0);
RWTexture2D<float4> tex2 : register(u1);

/*cbuffer cpCBvalues : register(b0) {
    uint frameCount;
};*/

[numthreads(2, 2, 1)]
void main(uint3 dtID : SV_DispatchThreadID) {

    int2 texsize;
    tex1.GetDimensions(texsize.x, texsize.y);
    int2 imageCoords = int2(dtID.xy);

    float4 c1 = tex1[imageCoords];
    float3 c2 = tex2[imageCoords].rgb;

    float4 c3 = float4(c2, 1);
    float4 cFinal = c1 * 0.9 + c3 * 0.1; //(c1 * (frameCount - 1) + c3) / frameCount;

    tex1[imageCoords] = cFinal;
}