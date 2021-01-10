const axios = require("axios")
const logger = require("node-color-log")
const fs = require("fs");

async function send(deadline, layers, counter) {
    const res = await axios.put(
        `htp://127.0.0.1:8080/predictions/googlenet/${deadline}/${layers}`,
         fs.readFileSync(__dirname + '/kitten_small.jpg'),{
            headers: {
              'Content-Type': "jpg"
            }
          });
    logger.info(`Request ${counter}:`, res.data)
}

async function sleep(ms) {
    await new Promise(resolve => setTimeout(resolve, ms));
    return;
}

async function main() {
    const configs = [
        // "0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1",
        // "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2",
        "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
        // "1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2",
    ]

    const rounds = 10;
    const iter = 15;

    for(let round = 0; round < rounds; round++) {
        for(let i = 0; i < iter; i++) {
            const deadline = 100000000; // 100ms
            const layers = configs[i % configs.length];
            send(deadline, layers, round * iter + i);
        }
        await sleep(50); // sleep 100ms
    }

}

main();
