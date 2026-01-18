import React from 'react';
import { Gauge, Database, Terminal, Cloud } from 'lucide-react';

export default function HardwareMonitor({ stats }) {
  const monitors = [
    { label: 'CPU', value: stats.cpu, icon: Gauge, color: '#00d9ff' },
    { label: 'RAM', value: stats.ram, icon: Database, color: '#b537f2' },
    { label: 'GPU', value: stats.gpu, icon: Terminal, color: '#ff2e97' },
    { label: 'VRAM', value: stats.vram, icon: Cloud, color: '#eab308' }
  ];

  return (
    <div className="p-6 border-b border-cyber-border">
      <h3 className="text-sm font-semibold text-cyber-muted uppercase tracking-wider mb-4">
        System Monitor
      </h3>
      <div className="space-y-3">
        {monitors.map(stat => {
          const Icon = stat.icon;
          return (
            <div key={stat.label}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <Icon size={14} style={{ color: stat.color }} />
                  <span className="text-xs font-medium text-cyber-muted">{stat.label}</span>
                </div>
                <span className="text-xs font-bold">{stat.value}%</span>
              </div>
              <div className="h-1.5 bg-cyber-card rounded-full overflow-hidden">
                <div
                  className="h-full transition-all duration-300"
                  style={{ width: `${stat.value}%`, backgroundColor: stat.color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
