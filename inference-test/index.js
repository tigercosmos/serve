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

async function main() {
    for(let i = 0; i < 10; i++) {
        const deadline = 10000000;
        const layers = "0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1";
        send(deadline, layers, i);
    }
}

main();
