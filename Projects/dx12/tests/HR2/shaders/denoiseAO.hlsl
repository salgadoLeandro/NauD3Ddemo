RWTexture2D<float> rtao : register(u0);
//RWTexture2D<float> depth : register(u1);
RWTexture2D<float4> positions : register(u1);
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

    float res = rtao[dtID];
    int total = 1;

    if (pud <= maxDist) { res += rtao[up]; ++total; }
    if (pdd <= maxDist) { res += rtao[down]; ++total; }
    if (pld <= maxDist) { res += rtao[left]; ++total; }
    if (prd <= maxDist) { res += rtao[right]; ++total; }
    if (puld <= maxDist) { res += rtao[upleft]; ++total; }
    if (purd <= maxDist) { res += rtao[upright]; ++total; }
    if (pdld <= maxDist) { res += rtao[downleft]; ++total; }
    if (pdrd <= maxDist) { res += rtao[downright]; ++total; }

    float rt = res / float(total);
    outp[dtID] = float4(rt, rt, rt, 1);
}

/*
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

    float dc = depth[dtID];
    float du = abs(depth[up] - dc);
    float dd = abs(depth[down] - dc);
    float dl = abs(depth[left] - dc);
    float dr = abs(depth[right] - dc);
    float dul = abs(depth[upleft] - dc);
    float dur = abs(depth[upright] - dc);
    float ddl = abs(depth[downleft] - dc);
    float ddr = abs(depth[downright] - dc);

    float res = rtao[dtID];

    uint total = 0;

    if (du <= maxDepth) { res += rtao[up] * du; ++total; }
    if (dd <= maxDepth) { res += rtao[down] * dd; ++total; }
    if (dl <= maxDepth) { res += rtao[left] * dl; ++total; }
    if (dr <= maxDepth) { res += rtao[right] * dr; ++total; }
    if (dul <= maxDepth) { res += rtao[upleft] * dul; ++total; }
    if (dur <= maxDepth) { res += rtao[upright] * dur; ++total; }
    if (ddl <= maxDepth) { res += rtao[downleft] * ddl; ++total; }
    if (ddr <= maxDepth) { res += rtao[downright] * ddr; ++total; }

    float rt = res / float(total);
    outp[dtID] = float4(rt, rt, rt, 1);
}
*/

