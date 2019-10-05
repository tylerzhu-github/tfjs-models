import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs-core';

export type BodyPixInput =
    ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|tf.Tensor3D;


export type PersonSegmentation = {
  data: Uint8Array,
  width: number,
  height: number,
  partData?: Int32Array,
  pose?: Pose,
};

export type PartSegmentation = {
  data: Int32Array,
  width: number,
  height: number,
  pose?: Pose,
};

export declare interface Padding {
  top: number, bottom: number, left: number, right: number
}

export declare type Part = {
  heatmapX: number,
  heatmapY: number,
  id: number
};

export declare type Vector2D = {
  y: number,
  x: number
};

export type TensorBuffer3D = tf.TensorBuffer<tf.Rank.R3>;


export declare type Color = {
  r: number,
  g: number,
  b: number,
  a: number,
};

export declare type Pose = posenet.Pose;
