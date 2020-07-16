#version 430

uniform mat4 m_pvm;

in vec4 position;
in vec4 texCoord0;

out Data {
    vec4 position;
    vec2 texCoord;
} DataOut;

void main() {
    DataOut.position = vec4(m_pvm * position);
    DataOut.texCoord = vec2(texCoord0.xy);
    gl_Position = DataOut.position;
}