//
//  ContentView.swift
//  EdgeAITask2
//
//

import SwiftUI
import PhotosUI
import Vision
import CoreML

// MARK: - Face Analyzer
final class FaceAnalyzer {

    // Cache models (loaded once)
    private let ageVNModel: VNCoreMLModel
    private let genderVNModel: VNCoreMLModel
    private let emotionVNModel: VNCoreMLModel

    init() {
        do {
            let age = try AgeNet(configuration: MLModelConfiguration())
            let gender = try GenderNet(configuration: MLModelConfiguration())
            let emotion = try CNNEmotions(configuration: MLModelConfiguration())

            self.ageVNModel = try VNCoreMLModel(for: age.model)
            self.genderVNModel = try VNCoreMLModel(for: gender.model)
            self.emotionVNModel = try VNCoreMLModel(for: emotion.model)
        } catch {
            fatalError("Failed to load CoreML models: \(error)")
        }
    }

    struct Output {
        let age: String
        let gender: String
        let emotion: String
        let modelTimeMs: Double
        let status: String
    }

    func analyze(uiImage: UIImage, completion: @escaping (Output) -> Void) {

        let totalStart = CFAbsoluteTimeGetCurrent()

        guard let cgImage = uiImage.cgImage else {
            completion(Output(age: "—", gender: "—", emotion: "—",
                              modelTimeMs: 0,
                              status: "Could not read image."))
            return
        }

        let request = VNDetectFaceRectanglesRequest { req, err in
            if let err = err {
                let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000
                completion(Output(age: "—", gender: "—", emotion: "—",
                                  modelTimeMs: totalMs,
                                  status: "Face detection error: \(err.localizedDescription)"))
                return
            }


            let inputCG: CGImage
            if let faces = req.results as? [VNFaceObservation],
               let face = faces.first,
               let lightlyCropped = self.cropFace(from: cgImage, faceObservation: face) {
                inputCG = lightlyCropped
            } else {
                inputCG = cgImage
            }

            DispatchQueue.global(qos: .userInitiated).async {
                let age = self.runAgeModel(inputCG)
                let gender = self.runGenderModel(inputCG)
                let emotion = self.runEmotionModel(inputCG)

                let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000

                completion(Output(age: age, gender: gender, emotion: emotion,
                                  modelTimeMs: totalMs,
                                  status: "Done"))
            }
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000
                completion(Output(age: "—", gender: "—", emotion: "—",
                                  modelTimeMs: totalMs,
                                  status: "Vision request failed: \(error.localizedDescription)"))
            }
        }
    }

    private func cropFace(from cgImage: CGImage, faceObservation: VNFaceObservation) -> CGImage? {
        let bbox = faceObservation.boundingBox
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)

        let x = bbox.origin.x * width
        let y = (1 - bbox.origin.y - bbox.size.height) * height
        let w = bbox.size.width * width
        let h = bbox.size.height * height

        // Light padding
        // Since UTK Dataset images are already cropped
        let paddingX = w * 0.08
        let paddingY = h * 0.10

        let newX = max(0, x - paddingX)
        let newY = max(0, y - paddingY)
        let newWidth = min(width - newX, w + 2 * paddingX)
        let newHeight = min(height - newY, h + 2 * paddingY)

        let rect = CGRect(x: newX, y: newY, width: newWidth, height: newHeight).integral
        return cgImage.cropping(to: rect)
    }

    private func runAgeModel(_ cgImage: CGImage) -> String {
        do {
            let request = VNCoreMLRequest(model: ageVNModel)
            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try handler.perform([request])

            guard let result = request.results?.first as? VNClassificationObservation else {
                return "Age not detected"
            }

            let label = result.identifier.trimmingCharacters(in: .whitespacesAndNewlines)

            // Binary mapping for age
            // Both 48-53 and 60-100 considered as Elderly
            if label.contains("48-53") || label.contains("60-100") {
                return "Elderly"
            } else {
                return "Adult"
            }

        } catch {
            return "Age model error"
        }
    }

    private func runGenderModel(_ cgImage: CGImage) -> String {
        do {
            let request = VNCoreMLRequest(model: genderVNModel)
            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try handler.perform([request])

            guard let result = request.results?.first as? VNClassificationObservation else {
                return "Gender not detected"
            }

            let label = result.identifier.lowercased()

            if label.contains("female") || label.contains("woman") {
                return "Woman"
            } else if label.contains("male") || label.contains("man") {
                return "Man"
            } else {
                return result.identifier
            }
        } catch {
            return "Gender model error"
        }
    }

    private func runEmotionModel(_ cgImage: CGImage) -> String {
        do {
            let request = VNCoreMLRequest(model: emotionVNModel)
            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try handler.perform([request])

            guard let result = request.results?.first as? VNClassificationObservation else {
                return "Emotion not detected"
            }

            let label = result.identifier.lowercased()

            return label.contains("happy") ? "Happy" : "Sad"
        } catch {
            return "Emotion model error"
        }
    }
}

// MARK: - UI
struct ContentView: View {

    private let analyzer = FaceAnalyzer()

    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?

    @State private var predictedAge = "Not predicted yet"
    @State private var predictedGender = "Not predicted yet"
    @State private var predictedEmotion = "Not predicted yet"

    @State private var modelTimeMs: Double?

    @State private var isAnalyzing = false
    @State private var statusMessage: String?

    var body: some View {
        VStack(spacing: 16) {

            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 320)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.gray.opacity(0.15))
                    .frame(height: 320)
                    .overlay(Text("No image selected").foregroundColor(.secondary))
                    .padding(.horizontal)
            }

            PhotosPicker("Select Photo", selection: $selectedItem, matching: .images)
                .buttonStyle(.borderedProminent)

            Button {
                guard let uiImage = selectedImage else { return }
                isAnalyzing = true
                statusMessage = "Analyzing…"

                analyzer.analyze(uiImage: uiImage) { output in
                    DispatchQueue.main.async {
                        self.predictedAge = output.age
                        self.predictedGender = output.gender
                        self.predictedEmotion = output.emotion
                        self.modelTimeMs = output.modelTimeMs
                        self.statusMessage = output.status
                        self.isAnalyzing = false
                    }
                }
            } label: {
                if isAnalyzing {
                    HStack(spacing: 8) {
                        ProgressView()
                        Text("Analyzing…")
                    }
                } else {
                    Text("Analyze Face")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(selectedImage == nil || isAnalyzing)

            VStack(alignment: .leading, spacing: 10) {
                Text("Age: \(predictedAge)").font(.title3).bold()
                Text("Gender: \(predictedGender)").font(.title3).bold()
                Text("Expression: \(predictedEmotion)").font(.title3).bold()

                if let m = modelTimeMs {
                    Text(String(format: "Model inference time: %.2f ms", m))
                        .font(.footnote)
                        .foregroundColor(.secondary)
                } else {
                    Text("Inference time: Not measured yet")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)

            if let msg = statusMessage {
                Text(msg)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
                    .multilineTextAlignment(.leading)
            }

            Spacer()
        }
        .padding()
        .onChange(of: selectedItem) {
            guard let item = selectedItem else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {

                    selectedImage = uiImage

                    predictedAge = "Not predicted yet"
                    predictedGender = "Not predicted yet"
                    predictedEmotion = "Not predicted yet"

                    modelTimeMs = nil
                    statusMessage = nil
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
