/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */
'use client';

import { createContext } from 'react';

import type { UITask } from '#modules/messages/types.ts';

export const TasksContext = createContext<TasksContextValue>({ tasks: [] });

export interface TasksContextValue {
  tasks: UITask[];
}
