const app = getApp();

Page({
  data: {
    devicePosition: "front",
    flashMode: "off",
    connected: false,
    streaming: false,
    stats: {
      fps: 0,
      detectionCount: 0
    }
  },
  onReady() {
    console.log('[onReady] Page ready, waiting for layout...');
    
    // Dr. Chen: 增加 500ms 延时，解决真机 Canvas 节点未就绪的问题
    setTimeout(() => {
        this.cameraContext = wx.createCameraContext();
        this.lastFrameTs = 0;
        this.framesSent = 0;
        this.overlayReady = false;
        this.encoderReady = false;
        this.canvasReady = false;
        
        // 初始化 Canvas
        this._initOverlayCanvas();
        this._initEncoderCanvas();
        
        // 连接 Socket
        this._connectSocket();
        
        console.log('[onReady] Initialization triggered after delay.');
        
        // Dr. Chen: 尝试自动启动流 (可选，方便调试)
        // this._startStreaming(); 
    }, 500);
  },
  onUnload() {
    this._stopStreaming();
    this._closeSocket();
  },
  _initOverlayCanvas() {
    wx.createSelectorQuery()
      .in(this)
      .select("#overlay")
      .fields({ node: true, size: true })
      .exec(res => {
        if (!res || !res[0]) return;
        const { node, width, height } = res[0];
        this.overlayCanvasNode = node;
        this.overlayCtx = node.getContext("2d");
        node.width = width;
        node.height = height;
        this.overlayReady = true;
        this.canvasReady = true;
      });
  },
  _initEncoderCanvas() {
    wx.createSelectorQuery()
      .in(this)
      .select("#frameEncoder")
      .fields({ node: true })
      .exec(res => {
        if (!res || !res[0]) return;
        const { node } = res[0];
        this.encoderCanvasNode = node;
        this.encoderCtx = node.getContext("2d");
        this.encoderReady = true;
      });
  },
  _connectSocket() {
    if (this.socket) return;
    // 修改为你的实际电脑局域网IP和8000端口
    const url = app?.globalData?.apiBaseUrl || "ws://192.168.1.4:5000/ws";
    this.socket = wx.connectSocket({ url });
    this.socket.onOpen(() => {
      this.setData({ connected: true });
    });
    this.socket.onClose(() => {
      this.setData({ connected: false });
      this.socket = null;
    });
    this.socket.onError(() => {
      this.setData({ connected: false });
    });
    this.socket.onMessage(evt => {
      try {
        const payload = JSON.parse(evt.data);
        
        // Dr. Chen 修正：把整个 payload 传进去，而不仅是 detections
        this._renderDetections(payload); 
        
        this.setData({
          stats: {
            fps: payload.fps || this.data.stats.fps,
            detectionCount: (payload.detections || []).length
          }
        });
      } catch (err) {
        console.warn("Invalid detection payload", err);
      }
    });
  },
  _closeSocket() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  },
  _renderDetections(payload) {
    // 注意：参数改为 payload，包含 detections 和 image_shape
    const detections = payload.detections || [];
    const imageShape = payload.image_shape || [853, 480]; // 默认防错 [高, 宽]
    
    if (!this.overlayReady || !this.overlayCtx || !this.overlayCanvasNode) return;
    
    const ctx = this.overlayCtx;
    const canvas = this.overlayCanvasNode;
    
    // 1. 获取比例因子
    // imageShape 来自后端 Python (H, W, C)，所以 0是高，1是宽
    const frameHeight = imageShape[0];
    const frameWidth = imageShape[1];
    
    const scaleX = canvas.width / frameWidth;
    const scaleY = canvas.height / frameHeight;

    // 2. 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 3. 绘制
    detections.forEach(item => {
      // 获取原始坐标
      let [x1, y1, x2, y2] = item.bbox || [];
      
      // 执行缩放 (关键步骤！)
      x1 = x1 * scaleX;
      y1 = y1 * scaleY;
      x2 = x2 * scaleX;
      y2 = y2 * scaleY;
      
      const width = x2 - x1;
      const height = y2 - y1;

      // 根据类别选择颜色
      const isMask = item.class_name === 'mask';
      const color = isMask ? "#22c55e" : "#ef4444"; // 绿 vs 红

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.font = "16px sans-serif";
      
      // 画框
      ctx.strokeRect(x1, y1, width, height);
      
      // 画标签背景
      const label = `${item.class_name} ${(item.confidence || 0).toFixed(2)}`;
      const textWidth = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 20, textWidth, 20);
      
      // 画文字
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 4, y1 - 6);
    });
  },
  toggleStreaming() {
    console.log('[toggleStreaming] Current streaming state:', this.data.streaming);
    console.log('[toggleStreaming] CameraContext:', this.cameraContext);
    console.log('[toggleStreaming] Canvas ready:', this.canvasReady);
    console.log('[toggleStreaming] Socket state:', this.socket ? this.socket.readyState : 'no socket');
    
    if (this.data.streaming) {
      this._stopStreaming();
    } else {
      this._startStreaming();
    }
  },
  _startStreaming() {
    console.log('[_startStreaming] Attempting to start streaming...');
    if (this.data.streaming) {
      console.log('[_startStreaming] Already streaming, skipping');
      return;
    }
    if (!this.cameraContext) {
      console.error('[_startStreaming] No camera context available');
      wx.showToast({
        title: '摄像头未初始化',
        icon: 'error'
      });
      return;
    }
    if (!this.overlayReady) {
      console.warn('[_startStreaming] Overlay canvas not ready, reinitializing...');
      this._initOverlayCanvas();
    }
    
    console.log('[_startStreaming] Setting up camera frame listener...');
    this.listener = this.cameraContext.onCameraFrame(frame => {
      const now = Date.now();
      const interval = 1000 / (app?.globalData?.frameRate || 5);
      if (now - this.lastFrameTs < interval) return;
      this.lastFrameTs = now;
      console.log('[Frame] Captured at', now, 'size:', frame.width, 'x', frame.height);
      this._sendFrame(frame);
    });
    
    this.listener.start({
      success: () => {
        console.log('[_startStreaming] Listener started successfully');
        this.setData({ streaming: true });
        wx.showToast({
          title: '检测已启动',
          icon: 'success'
        });
      },
      fail: (err) => {
        console.error('[_startStreaming] Failed to start listener:', err);
        wx.showToast({
          title: '启动失败: ' + (err.errMsg || '未知错误'),
          icon: 'error',
          duration: 3000
        });
      }
    });
  },
  _stopStreaming() {
    if (this.listener) {
      this.listener.stop();
      this.listener = null;
    }
    this.setData({ streaming: false });
  },
  _sendFrame(frame) {
    if (!this.socket || this.socket.readyState !== 1) return;
    if (!this.encoderReady || !this.encoderCanvasNode || !this.encoderCtx) {
      console.warn('[_sendFrame] Encoder canvas not ready, attempting reinitialization');
      this._initEncoderCanvas();
      return;
    }
    const { width, height, data } = frame;
    const canvas = this.encoderCanvasNode;
    canvas.width = width;
    canvas.height = height;
    const ctx = this.encoderCtx;
    const clamped = new Uint8ClampedArray(data);
    let imageData;
    if (typeof ImageData === 'function') {
      imageData = new ImageData(clamped, width, height);
    } else if (ctx && typeof ctx.createImageData === 'function') {
      imageData = ctx.createImageData(width, height);
      imageData.data.set(clamped);
    } else {
      console.error('[_sendFrame] 当前环境不支持 ImageData');
      return;
    }
    ctx.putImageData(imageData, 0, 0);
    
    // Dr. Chen 修正：toDataURL 是同步方法，直接获取返回值
    // 第一个参数是图片格式，第二个是质量(0-1)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    
    // 只有拿到数据才发送
    if (dataUrl && dataUrl.length > 100) {
        this.socket.send({
            data: JSON.stringify({
                type: "frame",
                payload: dataUrl
            })
        });
    } else {
        console.error('Frame encoding failed');
    }
  },
  switchCamera() {
    const next = this.data.devicePosition === "front" ? "back" : "front";
    this.setData({ devicePosition: next });
  },
  handleCameraError(evt) {
    console.error("Camera error", evt.detail);
    wx.showToast({
      title: "摄像头不可用",
      icon: "error"
    });
  },
  handleCameraStop() {
    this._stopStreaming();
  }
});
