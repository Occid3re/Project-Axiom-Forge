/**
 * Terminal-style transmission log — atmospheric, not informational.
 */

import { useEffect, useRef } from 'react';

interface TransmissionLogProps {
  entries: string[];
}

export function TransmissionLog({ entries }: TransmissionLogProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries.length]);

  return (
    <div className="relative overflow-hidden rounded-b-xl">
      {/* Top fade */}
      <div className="absolute inset-x-0 top-0 h-6 bg-gradient-to-b from-[#0a0b0f] to-transparent z-10 pointer-events-none" />

      <div
        className="h-24 overflow-y-auto px-4 py-2 font-mono text-[10px] leading-5 space-y-0.5"
        style={{ scrollbarWidth: 'none' }}
      >
        {entries.map((e, i) => {
          const isRecent = i >= entries.length - 3;
          return (
            <div key={i} className="flex gap-2 items-start">
              <span className="text-gray-700 shrink-0 tabular-nums">
                {String(i).padStart(4, '0')}
              </span>
              <span
                className="transition-colors duration-1000"
                style={{ color: isRecent ? '#6ee7b7' : '#374151' }}
              >
                {e}
              </span>
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}
