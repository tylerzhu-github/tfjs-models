/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import {getBackend} from '@tensorflow/tfjs-core';

import {PartSegmentation, PersonSegmentation, Pose} from '../types';

import {partMasksForPosesCPU, personMasksForPosesCPU, personMasksForPosesGPU} from './person_masks_for_poses';

export function toPersonKSegmentation(
    segmentation: tf.Tensor2D, k: number): tf.Tensor2D {
  return tf.tidy(
      () => (segmentation.equal(tf.scalar(k)).toInt() as tf.Tensor2D));
}

export function toPersonKPartSegmentation(
    segmentation: tf.Tensor2D, bodyParts: tf.Tensor2D, k: number): tf.Tensor2D {
  return tf.tidy(
      () => (segmentation.equal(tf.scalar(k)).toInt().mul(bodyParts.add(1)))
                .sub(1) as tf.Tensor2D);
}

function isWebGlBackend() {
  return getBackend() === 'webgl';
}

export async function decodePersonSegmentationMasksForPoses(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D, poses: Pose[],
    height: number, width: number, stride: number,
    [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    minPoseScore = 0.2, refineSteps = 8, minKeypointScore = 0.3,
    maxNumPeople = 10): Promise<PersonSegmentation[]> {
  // Filter out poses with smaller score.
  const posesAboveScore = poses.filter(pose => pose.score >= minPoseScore);

  let personSegmentationsData: Uint8Array[];

  if (isWebGlBackend()) {
    const personSegmentations = tf.tidy(() => {
      const masksTensor = personMasksForPosesGPU(
          segmentation, longOffsets, posesAboveScore, height, width, stride,
          [inHeight, inWidth], [[padT, padB], [padL, padR]], refineSteps,
          minKeypointScore, maxNumPeople);

      return posesAboveScore.map(
          (_, k) => toPersonKSegmentation(masksTensor, k));
    });

    personSegmentationsData =
        (await Promise.all(personSegmentations.map(mask => mask.data())) as
         Uint8Array[]);

    personSegmentations.forEach(x => x.dispose());
  } else {
    const segmentationsData = await segmentation.data() as Uint8Array;
    const longOffsetsData = await longOffsets.data() as Float32Array;

    personSegmentationsData = personMasksForPosesCPU(
        segmentationsData, longOffsetsData, posesAboveScore, height, width,
        stride, [inHeight, inWidth], [[padT, padB], [padL, padR]], refineSteps);
  }

  return personSegmentationsData.map(
      (data, i) => ({data, pose: posesAboveScore[i], width, height}));
}

export async function decodePartMasksForPoses(
    segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D,
    partSegmentation: tf.Tensor2D, poses: Pose[], height: number, width: number,
    stride: number, [inHeight, inWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    minPoseScore = 0.2, refineSteps = 8, minKeypointScore = 0.3,
    maxNumPeople = 10): Promise<PartSegmentation[]> {
  // Filter out poses with smaller score.
  const posesAboveScore = poses.filter(pose => pose.score >= minPoseScore);

  let partSegmentationsByPersonData: Int32Array[];

  if (isWebGlBackend()) {
    const partSegmentations = tf.tidy(() => {
      const masksTensor = personMasksForPosesGPU(
          segmentation, longOffsets, posesAboveScore, height, width, stride,
          [inHeight, inWidth], [[padT, padB], [padL, padR]], refineSteps,
          minKeypointScore, maxNumPeople);

      return posesAboveScore.map(
          (_, k) =>
              toPersonKPartSegmentation(masksTensor, partSegmentation, k));
    });

    partSegmentationsByPersonData =
        (await Promise.all(partSegmentations.map(x => x.data()))) as
        Int32Array[];

    partSegmentations.forEach(x => x.dispose());
  } else {
    const segmentationsData = await segmentation.data() as Uint8Array;
    const longOffsetsData = await longOffsets.data() as Float32Array;
    const partSegmentaionData = await partSegmentation.data() as Uint8Array;

    partSegmentationsByPersonData = partMasksForPosesCPU(
        segmentationsData, longOffsetsData, partSegmentaionData,
        posesAboveScore, height, width, stride, [inHeight, inWidth],
        [[padT, padB], [padL, padR]], refineSteps);
  }

  return partSegmentationsByPersonData.map(
      (data, k) => ({pose: posesAboveScore[k], data, height, width}));
}
