import 'dotenv/config';
import { createApp } from './app.js';

const PORT = process.env.PORT || 4500;

createApp().listen(PORT, () => {
  console.log(`documind-api listening on http://localhost:${PORT}`);
});
