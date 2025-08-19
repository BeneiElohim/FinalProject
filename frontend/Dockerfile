# --- Stage 1: Build Dependencies ---
FROM node:18 AS builder
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the source code
COPY . .

# Build the static production assets
RUN npm run build

# --- Stage 2: Production Server ---
FROM nginx:1.27-alpine

# Copy the built assets from the 'builder' stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Tell Docker that the container listens on port 80
EXPOSE 80

# The default nginx command is fine, but we make it explicit.
CMD ["nginx", "-g", "daemon off;"]