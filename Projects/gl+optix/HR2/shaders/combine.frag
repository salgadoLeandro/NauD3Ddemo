#version 430

uniform sampler2D gl;
uniform sampler2D mask;
uniform sampler2D optx;
uniform sampler2D rtao;
uniform sampler2D rts;

in Data {
    vec4 position;
    vec2 texCoord;
} DataIn;

out vec4 color;

void main() {
    
    vec4 r1 = texture(gl, DataIn.texCoord);
    vec4 mk = texture(mask, DataIn.texCoord);
    vec4 r2 = texture(optx, DataIn.texCoord);
    vec4 ao = texture(rtao, DataIn.texCoord);
    vec4 sh = texture(rts, DataIn.texCoord);

    vec4 aot = vec4(pow(ao.rgb, vec3(1.0f / 3.0f)), 1.0f);
    vec4 sht = vec4(pow(sh.rgb, vec3(1.0f / 3.0f)), 1.0f);
    color = mk.r == 1.0f ? r2 : r1 * sht * aot;
    
}