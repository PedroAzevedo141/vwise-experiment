import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_v2_l


class ImageClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Crie uma instância do modelo
        self.model = efficientnet_v2_l(weights=None)

        # Carregue o estado do modelo
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


if __name__ == "__main__":

    gt_dict = {
        0: "Dirt or service road",
        1: "Factory interior",
        2: "High-density urban avenue",
        3: "Highway",
        4: "Low-density urban street",
        5: "Port or dock area",
    }

    classifier = ImageClassifier(r"pretrained\model_efficient_91.pt")
    image_path = r"./../assets/dirt_road.jpeg"
    predicted_class = classifier.predict(image_path)
    print(f"Predicted class: {gt_dict[predicted_class]}")

    # -=-=-

    image_path = r"./../assets/high_density.jpeg"
    predicted_class = classifier.predict(image_path)
    print(f"Predicted class: {gt_dict[predicted_class]}")

    # -=-=-

    image_path = r"./../assets/highway.jpg"
    predicted_class = classifier.predict(image_path)
    print(f"Predicted class: {gt_dict[predicted_class]}")

    # -=-=-

    image_path = r"./../assets/low_density.jpeg"
    predicted_class = classifier.predict(image_path)
    print(f"Predicted class: {gt_dict[predicted_class]}")
