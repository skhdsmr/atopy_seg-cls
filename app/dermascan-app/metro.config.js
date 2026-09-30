// Metro 설정: .tflite 모델을 정적 asset 으로 번들해서 require() 로 로드할 수 있게 한다.
// (react-native-fast-tflite 가 require('...tflite') 에셋을 받는다)
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('tflite');

module.exports = config;
