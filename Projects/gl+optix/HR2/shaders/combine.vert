#version 430

in vec4 position;
in vec4 texCoord0;

out Data {
    vec4 position;
    vec2 texCoord;
} DataOut;

void main() {
    DataOut.position = position;
    DataOut.texCoord = vec2(texCoord0.xy);
    gl_Position = DataOut.position;
}