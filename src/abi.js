// Compact KDoc/operator ABI shared by the KX reader and player runtime.
// These values mirror the flags and output classes in the released kdoc and
// player demo sources; keeping one owner prevents parser/runtime drift.
export const OPC_BLOB = 0x00080000;
export const OPC_KENV = 0x00800000;
export const OPC_VARIABLEINPUT = 0x01000000;
export const OPC_SKIPEXEC = 0x02000000;
export const OPC_ALTEXEC = 0x04000000;
export const OPC_STRIPPEDIN = 0x08000000;
export const OPC_DONTCALLLINK = 0x10000000;
export const OPC_ALTINIT = 0x20000000;
export const OPC_KOP = 0x40000000;
export const OPC_FLEXINPUT = 0x80000000;

export const KC_BITMAP = 1;
export const KC_MINMESH = 2;
export const KC_SCENE = 3;
export const KC_MATERIAL = 5;
export const KC_MESH = 6;
export const KC_IPP = 8;
export const KC_EFFECT = 9;
export const KC_DEMO = 11;
export const KC_SPLINE = 13;
export const KC_ANY = 255;
