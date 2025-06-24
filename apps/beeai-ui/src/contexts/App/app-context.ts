/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, type Dispatch, type SetStateAction } from 'react';

export const AppContext = createContext<AppContextValue>({});

interface AppContextValue {
  navigationOpen?: boolean;
  agentDetailOpen?: boolean;
  closeNavOnClickOutside?: boolean;
  setNavigationOpen?: Dispatch<SetStateAction<boolean>>;
  setAgentDetailOpen?: Dispatch<SetStateAction<boolean>>;
  setCloseNavOnClickOutside?: Dispatch<SetStateAction<boolean>>;
}
