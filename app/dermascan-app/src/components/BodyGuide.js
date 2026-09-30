import React from 'react';
import Svg, { Path, Ellipse, G } from 'react-native-svg';

// Body part name -> shape category
function shapeOf(part = '') {
  const p = part;
  if (p.includes('손')) return 'hand';
  if (p.includes('발')) return 'foot';
  if (p.includes('얼굴') || p.includes('머리') || p.includes('뒤통수') || p.includes('목')) return 'face';
  if (p.includes('상완') || p.includes('전완') || p.includes('어깨')) return 'arm';
  if (p.includes('허벅지') || p.includes('정강이') || p.includes('종아리')) return 'leg';
  if (p.includes('가슴') || p.includes('복부') || p.includes('등') || p.includes('허리') || p.includes('엉덩이')) return 'torso';
  return 'oval';
}

const STROKE = 'rgba(255,255,255,0.92)';
const SW = 2.2;

// Continuous silhouette paths (viewBox 0 0 100 140) — fill the screen generously
const PATHS = {
  // Palm + four finger humps + thumb
  hand:
    'M26 120 Q22 96 27 78 Q16 76 13 88 Q11 96 19 95 Q27 92 31 80 ' +
    'Q31 44 37 42 Q43 44 44 74 Q45 44 51 42 Q57 44 58 74 ' +
    'Q59 46 65 44 Q71 46 72 74 Q73 50 78 48 Q83 50 82 78 ' +
    'Q86 98 80 120 Z',
  // Arm, thick at top and tapering downward
  arm:
    'M50 8 C66 8 66 26 63 44 C61 72 59 98 57 120 C57 132 43 132 43 120 ' +
    'C41 98 39 72 37 44 C34 26 34 8 50 8 Z',
  // Leg (longer and thicker than the arm)
  leg:
    'M50 6 C68 6 69 28 65 50 C62 80 60 105 58 126 C58 138 42 138 42 126 ' +
    'C40 105 38 80 35 50 C31 28 32 6 50 6 Z',
  // Foot (instep to toes, rounded heel)
  foot:
    'M46 14 C60 12 60 38 62 56 C64 78 72 90 72 106 C72 124 54 126 46 120 ' +
    'C36 115 34 96 34 74 C33 50 32 28 38 18 C40 14 43 14 46 14 Z',
  // Torso (shoulders down to waist)
  torso:
    'M22 26 C22 18 32 16 40 19 C45 21 55 21 60 19 C68 16 78 18 78 26 ' +
    'C76 44 73 76 67 110 C64 122 36 122 33 110 C27 76 24 44 22 26 Z',
};

function Shape({ kind }) {
  const common = { stroke: STROKE, strokeWidth: SW, fill: 'rgba(255,255,255,0.05)', strokeLinejoin: 'round', strokeLinecap: 'round' };
  if (kind === 'face') return <Ellipse cx="50" cy="64" rx="33" ry="46" {...common} />;
  if (kind === 'oval') return <Ellipse cx="50" cy="70" rx="36" ry="56" {...common} />;
  return <G {...common}><Path d={PATHS[kind]} /></G>;
}

export default function BodyGuide({ bodyPart, style }) {
  return (
    <Svg viewBox="0 0 100 140" style={style}>
      <Shape kind={shapeOf(bodyPart)} />
    </Svg>
  );
}
