package org.pytorch.serve;

import org.pytorch.serve.ModelServer;
import org.pytorch.serve.Scheduler;
import org.pytorch.serve.util.logging.QLogLayout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SchedulerThread implements Runnable {
    private Logger logger = LoggerFactory.getLogger(SchedulerThread.class);

    private Scheduler scheduler;

    public SchedulerThread(Scheduler s) {
        scheduler = s;
    }

    @Override
    public void run() {
        logger.info("XXXXXXXXXXOOOOO Scheduler has {} models {} GPU workers", scheduler.getModelNumber(), scheduler.getGpuNumber());
        while(true) {
            int interval = 5; //ms
            long current = System.nanoTime();
            scheduler.schedule(current + interval * 1000000);
            try
            {
                Thread.sleep(5);
            }
            catch(InterruptedException ex)
            {
                Thread.currentThread().interrupt();
            }
        }
    }
}