RWTexture2D<float4> positions : register(u0);
RWTexture2D<float4> comb : register(u1);
RWTexture2D<float4> outp : register(u2);

static const float maxDepth = 0.5f;
static const float maxDist = 2.0f;

[numthreads(2, 2, 1)]
void main(uint2 dtID : SV_DispatchThreadID) {

    uint2 up = dtID + uint2(0, -1);
    uint2 down = dtID + uint2(0, 1);
    uint2 left = dtID + uint2(-1, 0);
    uint2 right = dtID + uint2(1, 0);
    uint2 upleft = dtID + uint2(-1, -1);
    uint2 upright = dtID + uint2(1, -1);
    uint2 downleft = dtID + uint2(-1, 1);
    uint2 downright = dtID + uint2(1, 1);

    float4 pc = positions[dtID];
    float4 pu = positions[up];
    float4 pd = positions[down];
    float4 pl = positions[left];
    float4 pr = positions[right];
    float4 pul = positions[upleft];
    float4 pur = positions[upright];
    float4 pdl = positions[downleft];
    float4 pdr = positions[downright];

    float pud = length(pc - pu);
    float pdd = length(pc - pd);
    float pld = length(pc - pl);
    float prd = length(pc - pr);
    float puld = length(pc - pul);
    float purd = length(pc - pur);
    float pdld = length(pc - pdl);
    float pdrd = length(pc - pdr);

    float4 res = comb[dtID];;

    uint total = 1;

    if (pud <= maxDist) { res += comb[up]; ++total; }
    if (pdd <= maxDist) { res += comb[down]; ++total; }
    if (pld <= maxDist) { res += comb[left]; ++total; }
    if (prd <= maxDist) { res += comb[right]; ++total; }
    if (puld <= maxDist) { res += comb[upleft]; ++total; }
    if (purd <= maxDist) { res += comb[upright]; ++total; }
    if (pdld <= maxDist) { res += comb[downleft]; ++total; }
    if (pdrd <= maxDist) { res += comb[downright]; ++total; }

    outp[dtID] = res / (float(total));
}