/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export function fillArray<T>(element: T, size: number): T[] {
  const result: T[] = new Array(size);

  for (let i = 0; i < size; i++) {
    result[i] = element;
  }

  return result;
}

export function clamp(a: number, min: number, max: number): number {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

export function squaredDistance(
    y1: number, x1: number, y2: number, x2: number): number {
  const dy = y2 - y1;
  const dx = x2 - x1;
  return dy * dy + dx * dx;
}
