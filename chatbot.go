package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"milvus/go_sdk/models"

	"github.com/google/uuid"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	// Collection configuration
	MessageCollectionName = "messages"
	MessageDimension      = 768 // Standard embedding dimension
	IndexType             = entity.IvfFlat
	MetricType            = entity.L2
	IndexParams           = `{"nlist": 1024}`
	SearchParams          = `{"nprobe": 10}`
)

type MessageStore struct {
	client client.Client
}

func NewMessageStore(milvusClient client.Client) *MessageStore {
	store := &MessageStore{client: milvusClient}

	// Initialize collection on startup
	if err := store.initializeCollection(context.Background()); err != nil {
		log.Printf("Failed to initialize Milvus collection: %v", err)
	}

	return store
}

func (s *MessageStore) initializeCollection(ctx context.Context) error {
	// Check if collection exists
	hasCollection, err := s.client.HasCollection(ctx, MessageCollectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if hasCollection {
		return nil // Collection already exists
	}

	// Define schema for messages collection
	schema := entity.NewSchema()
	schema.WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeVarChar).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("user_id").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("conversation_id").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("language").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("ai_message").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("ai_response").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("message").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("message_embedding").WithDataType(entity.FieldTypeFloatVector)).
		WithField(entity.NewField().WithName("respone").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("response_index").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("response_embedding").WithDataType(entity.FieldTypeFloatVector)).
		WithField(entity.NewField().WithName("feedback").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("metadata").WithDataType(entity.FieldTypeJSON)).
		WithField(entity.NewField().WithName("status").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("created_at").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("updated_at").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("deleted_at").WithDataType(entity.FieldTypeInt64))

	// Create collection
	err = s.client.CreateCollection(ctx, schema, entity.DefaultShardNumber)
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	// Create index for vector fields
	idxScalar := entity.NewScalarIndexWithType(entity.Sorted)
	s.client.CreateIndex(ctx, MessageCollectionName, "id", idxScalar, false)
	s.client.CreateIndex(ctx, MessageCollectionName, "user_id", idxScalar, false)
	s.client.CreateIndex(ctx, MessageCollectionName, "conversation_id", idxScalar, false)
	s.client.CreateIndex(ctx, MessageCollectionName, "message", idxScalar, false)
	s.client.CreateIndex(ctx, MessageCollectionName, "response", idxScalar, false)

	idxVector, err := entity.NewIndexIvfFlat(entity.IP, 1024)
	if err != nil {
		log.Fatal("failed to new index:", err.Error())
	}
	s.client.CreateIndex(ctx, MessageCollectionName, "message_embedding", idxVector, false)
	s.client.CreateIndex(ctx, MessageCollectionName, "response_embedding", idxVector, false)

	// Load collection
	err = s.client.LoadCollection(ctx, MessageCollectionName, false)
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	return nil
}

func (s *MessageStore) Create(ctx context.Context, message *models.Message) error {
	// Prepare data for insertion
	ids := []string{message.ID.String()}
	conversationIDs := []string{message.ConversationID.String()}
	messages := []string{message.Message}
	responses := []string{message.Response}
	statuses := []string{string(message.Status)}
	feedbacks := []string{string(message.Feedback)}
	createdAts := []int64{message.CreatedAt.Unix()}
	updatedAts := []int64{message.UpdatedAt.Unix()}

	// Handle metadata
	metadataBytes, err := json.Marshal(message.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}
	metadatas := [][]byte{metadataBytes}

	// Handle embeddings - convert string embeddings to float vectors
	var messageEmbeddings, responseEmbeddings [][]float32

	if message.MessageEmbedding != nil {
		msgEmb, err := s.parseEmbedding(*message.MessageEmbedding)
		if err != nil {
			return fmt.Errorf("failed to parse message embedding: %w", err)
		}
		messageEmbeddings = [][]float32{msgEmb}
	} else {
		// Use zero vector if no embedding provided
		messageEmbeddings = [][]float32{make([]float32, MessageDimension)}
	}

	if message.ResponseEmbedding != nil {
		respEmb, err := s.parseEmbedding(*message.ResponseEmbedding)
		if err != nil {
			return fmt.Errorf("failed to parse response embedding: %w", err)
		}
		responseEmbeddings = [][]float32{respEmb}
	} else {
		// Use zero vector if no embedding provided
		responseEmbeddings = [][]float32{make([]float32, MessageDimension)}
	}

	// Create columns for insertion
	columns := []entity.Column{
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnVarChar("conversation_id", conversationIDs),
		entity.NewColumnVarChar("message", messages),
		entity.NewColumnVarChar("response", responses),
		entity.NewColumnFloatVector("message_embedding", MessageDimension, messageEmbeddings),
		entity.NewColumnFloatVector("response_embedding", MessageDimension, responseEmbeddings),
		entity.NewColumnVarChar("status", statuses),
		entity.NewColumnVarChar("feedback", feedbacks),
		entity.NewColumnJSONBytes("metadata", metadatas),
		entity.NewColumnInt64("created_at", createdAts),
		entity.NewColumnInt64("updated_at", updatedAts),
	}

	// Insert data
	_, err = s.client.Insert(ctx, MessageCollectionName, "", columns...)
	if err != nil {
		return fmt.Errorf("failed to insert message: %w", err)
	}

	// Flush to ensure data is persisted
	err = s.client.Flush(ctx, MessageCollectionName, false)
	if err != nil {
		return fmt.Errorf("failed to flush collection: %w", err)
	}

	return nil
}

func (s *MessageStore) GetByConversationID(ctx context.Context, conversationID uuid.UUID, limit, offset int) ([]models.Message, error) {
	// Build query expression
	expr := fmt.Sprintf(`conversation_id == "%s"`, conversationID.String())

	// Define output fields
	outputFields := []string{"id", "conversation_id", "message", "response", "status", "feedback", "metadata", "created_at", "updated_at"}

	// Query with pagination
	queryResult, err := s.client.Query(ctx, MessageCollectionName, []string{}, expr, outputFields, client.WithLimit(int64(limit)), client.WithOffset(int64(offset)))
	if err != nil {
		return nil, fmt.Errorf("failed to query messages: %w", err)
	}

	return s.parseQueryResults(queryResult)
}

func (s *MessageStore) GetCountByConversationID(ctx context.Context, conversationID uuid.UUID) (int, error) {
	expr := fmt.Sprintf(`conversation_id == "%s"`, conversationID.String())

	queryResult, err := s.client.Query(ctx, MessageCollectionName, []string{}, expr, []string{"id"})
	if err != nil {
		return 0, fmt.Errorf("failed to count messages: %w", err)
	}

	if len(queryResult) == 0 {
		return 0, nil
	}

	// Get the first column (id) to count rows
	if idColumn := queryResult[0]; idColumn != nil {
		return idColumn.Len(), nil
	}

	return 0, nil
}

func (s *MessageStore) GetCountByUserID(ctx context.Context, userID uuid.UUID) (int, error) {
	// Note: This would require a separate query to get conversation IDs for the user first
	// For now, return 0 as this requires conversation store integration
	return 0, nil
}

func (s *MessageStore) DeleteByConversationID(ctx context.Context, conversationID uuid.UUID) error {
	expr := fmt.Sprintf(`conversation_id == "%s"`, conversationID.String())

	err := s.client.Delete(ctx, MessageCollectionName, "", expr)
	if err != nil {
		return fmt.Errorf("failed to delete messages: %w", err)
	}

	return nil
}

func (s *MessageStore) GetMessageStats(ctx context.Context, userID uuid.UUID) (*models.MessageStats, error) {
	// This would require aggregation queries which are complex in Milvus
	// For now, return basic stats
	stats := &models.MessageStats{
		TotalMessages:     0,
		AverageResponse:   0.0,
		AverageConfidence: 0.0,
	}
	return stats, nil
}

func (s *MessageStore) SearchMessages(ctx context.Context, userID uuid.UUID, req *models.MessageSearchRequest) ([]models.Message, error) {
	if req.Query == "" {
		// If no query, use regular filtering
		return s.filterMessages(ctx, req)
	}

	// For vector search, we would need the query embedding
	// This is a simplified implementation
	return s.filterMessages(ctx, req)
}

func (s *MessageStore) SearchSimilarMessages(ctx context.Context, queryEmbedding []float32, conversationID *uuid.UUID, limit int) ([]models.Message, error) {
	// Prepare search vectors
	searchVectors := []entity.Vector{entity.FloatVector(queryEmbedding)}

	// Build expression for filtering
	var expr string
	if conversationID != nil {
		expr = fmt.Sprintf(`conversation_id == "%s"`, conversationID.String())
	}

	// Define output fields
	outputFields := []string{"id", "conversation_id", "message", "response", "status", "feedback", "metadata", "created_at", "updated_at"}

	// Perform vector search
	searchResult, err := s.client.Search(ctx, MessageCollectionName, []string{}, expr, outputFields, searchVectors, "message_embedding", MetricType, limit /*SearchParams*/, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to search similar messages: %w", err)
	}

	if len(searchResult) == 0 {
		return []models.Message{}, nil
	}

	// Convert search results to messages
	return s.parseSearchResults(searchResult[0])
}

func (s *MessageStore) Update(ctx context.Context, message *models.Message) error {
	// Delete existing message
	expr := fmt.Sprintf(`id == "%s"`, message.ID.String())
	err := s.client.Delete(ctx, MessageCollectionName, "", expr)
	if err != nil {
		return fmt.Errorf("failed to delete existing message: %w", err)
	}

	// Insert updated message
	return s.Create(ctx, message)
}

// Helper functions

func (s *MessageStore) parseEmbedding(embeddingStr string) ([]float32, error) {
	// Assuming embedding is stored as JSON array string
	var embedding []float32
	err := json.Unmarshal([]byte(embeddingStr), &embedding)
	if err != nil {
		return nil, fmt.Errorf("failed to parse embedding: %w", err)
	}

	// Ensure correct dimension
	if len(embedding) != MessageDimension {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d", MessageDimension, len(embedding))
	}

	return embedding, nil
}

func (s *MessageStore) parseQueryResults(results []entity.Column) ([]models.Message, error) {
	if len(results) == 0 {
		return []models.Message{}, nil
	}

	// Get the number of rows from the first column
	rowCount := results[0].Len()
	messages := make([]models.Message, rowCount)

	// Parse each column
	for i := 0; i < rowCount; i++ {
		message := models.Message{}

		for _, column := range results {
			switch column.Name() {
			case "id":
				if idCol, ok := column.(*entity.ColumnVarChar); ok {
					uid, _ := idCol.ValueByIdx(i)
					id, err := uuid.Parse(uid)
					if err != nil {
						return nil, fmt.Errorf("failed to parse message ID: %w", err)
					}
					message.ID = id
				}
			case "conversation_id":
				if convCol, ok := column.(*entity.ColumnVarChar); ok {
					uid, _ := convCol.ValueByIdx(i)
					convID, err := uuid.Parse(uid)
					if err != nil {
						return nil, fmt.Errorf("failed to parse conversation ID: %w", err)
					}
					message.ConversationID = convID
				}
			case "message":
				if msgCol, ok := column.(*entity.ColumnVarChar); ok {
					message.Message, _ = msgCol.ValueByIdx(i)
				}
			case "response":
				if respCol, ok := column.(*entity.ColumnVarChar); ok {
					message.Response, _ = respCol.ValueByIdx(i)
				}
			case "status":
				if statusCol, ok := column.(*entity.ColumnVarChar); ok {
					s, _ := statusCol.ValueByIdx(i)
					message.Status = models.MessageStatus(s)
				}
			case "feedback":
				if feedbackCol, ok := column.(*entity.ColumnVarChar); ok {
					f, _ := feedbackCol.ValueByIdx(i)
					message.Feedback = models.UserFeedback(f)
				}
			case "metadata":
				if metaCol, ok := column.(*entity.ColumnJSONBytes); ok {
					var metadata models.Metadata
					m, _ := metaCol.ValueByIdx(i)
					err := json.Unmarshal(m, &metadata)
					if err != nil {
						return nil, fmt.Errorf("failed to parse metadata: %w", err)
					}
					message.Metadata = metadata
				}
			case "created_at":
				if createdCol, ok := column.(*entity.ColumnInt64); ok {
					c, _ := createdCol.ValueByIdx(i)
					message.CreatedAt = time.Unix(c, 0)
				}
			case "updated_at":
				if updatedCol, ok := column.(*entity.ColumnInt64); ok {
					u, _ := updatedCol.ValueByIdx(i)
					message.UpdatedAt = time.Unix(u, 0)
				}
			}
		}

		messages[i] = message
	}

	return messages, nil
}

func (s *MessageStore) parseSearchResults(result client.SearchResult) ([]models.Message, error) {
	messages := make([]models.Message, result.IDs.Len())

	// for i, id := range result.IDs {
	// 	message := models.Message{}

	// 	// Parse ID
	// 	if idStr, ok := id.(string); ok {
	// 		messageID, err := uuid.Parse(idStr)
	// 		if err != nil {
	// 			return nil, fmt.Errorf("failed to parse message ID: %w", err)
	// 		}
	// 		message.ID = messageID
	// 	}

	// 	// Parse fields from search result
	// 	for fieldName, fieldData := range result.Fields {
	// 		switch fieldName {
	// 		case "conversation_id":
	// 			if convIDs, ok := fieldData.([]string); ok && i < len(convIDs) {
	// 				convID, err := uuid.Parse(convIDs[i])
	// 				if err != nil {
	// 					return nil, fmt.Errorf("failed to parse conversation ID: %w", err)
	// 				}
	// 				message.ConversationID = convID
	// 			}
	// 		case "message":
	// 			if messages, ok := fieldData.([]string); ok && i < len(messages) {
	// 				message.Message = messages[i]
	// 			}
	// 		case "response":
	// 			if responses, ok := fieldData.([]string); ok && i < len(responses) {
	// 				message.Response = responses[i]
	// 			}
	// 		case "status":
	// 			if statuses, ok := fieldData.([]string); ok && i < len(statuses) {
	// 				message.Status = models.MessageStatus(statuses[i])
	// 			}
	// 		case "feedback":
	// 			if feedbacks, ok := fieldData.([]string); ok && i < len(feedbacks) {
	// 				message.Feedback = models.UserFeedback(feedbacks[i])
	// 			}
	// 		case "created_at":
	// 			if timestamps, ok := fieldData.([]int64); ok && i < len(timestamps) {
	// 				message.CreatedAt = time.Unix(timestamps[i], 0)
	// 			}
	// 		case "updated_at":
	// 			if timestamps, ok := fieldData.([]int64); ok && i < len(timestamps) {
	// 				message.UpdatedAt = time.Unix(timestamps[i], 0)
	// 			}
	// 		}
	// 	}

	// 	messages[i] = message
	// }

	return messages, nil
}

func (s *MessageStore) filterMessages(ctx context.Context, req *models.MessageSearchRequest) ([]models.Message, error) {
	var expressions []string

	// Build filter expressions
	if req.ConversationID != nil {
		expressions = append(expressions, fmt.Sprintf(`conversation_id == "%s"`, req.ConversationID.String()))
	}

	if req.Status != nil {
		expressions = append(expressions, fmt.Sprintf(`status == "%s"`, string(*req.Status)))
	}

	if req.Feedback != nil {
		expressions = append(expressions, fmt.Sprintf(`feedback == "%s"`, string(*req.Feedback)))
	}

	if req.StartDate != nil {
		expressions = append(expressions, fmt.Sprintf(`created_at >= %d`, req.StartDate.Unix()))
	}

	if req.EndDate != nil {
		expressions = append(expressions, fmt.Sprintf(`created_at <= %d`, req.EndDate.Unix()))
	}

	// Combine expressions with AND
	var expr string
	if len(expressions) > 0 {
		expr = expressions[0]
		for i := 1; i < len(expressions); i++ {
			expr += " && " + expressions[i]
		}
	}

	// Set default limit if not provided
	limit := req.Limit
	if limit <= 0 {
		limit = 100
	}

	// Define output fields
	outputFields := []string{"id", "conversation_id", "message", "response", "status", "feedback", "metadata", "created_at", "updated_at"}

	// Query with filters
	queryResult, err := s.client.Query(ctx, MessageCollectionName, []string{}, expr, outputFields, client.WithLimit(int64(limit)), client.WithOffset(int64(req.Offset)))
	if err != nil {
		return nil, fmt.Errorf("failed to filter messages: %w", err)
	}

	return s.parseQueryResults(queryResult)
}

func main() {
	fmt.Println("=== Milvus MessageStore Test ===")

	// Create a context
	ctx := context.Background()
	_ = ctx // Use context to avoid unused variable error

	// Note: In a real application, you would connect to an actual Milvus instance
	fmt.Println("Connecting to Milvus...")

	// Example Milvus client connection (uncomment when you have Milvus running)
	milvusClient, err := client.NewClient(ctx, client.Config{
		Address: "localhost:19530",
	})
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}
	defer milvusClient.Close()

	// For demonstration, we'll show the API usage
	fmt.Println("\n1. Creating MessageStore...")
	store := NewMessageStore(milvusClient)

	// Create sample data for testing
	conversationID := uuid.New()
	fmt.Printf("Using conversation ID: %s\n", conversationID.String())

	// Example of how to call GetByConversationID function:
	fmt.Println("\n2. Calling GetByConversationID function...")

	// Uncomment these lines when you have a real Milvus client:
	messages, err := store.GetByConversationID(ctx, conversationID, 10, 0)
	if err != nil {
		fmt.Printf("GetByConversationID error: %v\n", err)
	} else {
		fmt.Printf("Found %d messages in conversation\n", len(messages))
		for i, msg := range messages {
			fmt.Printf("Message %d: ID=%s, Content=%s\n", i+1, msg.ID.String(), msg.Message)
		}
	}

	// Parameters explanation:
	fmt.Println("\nGetByConversationID function parameters:")
	fmt.Println("- ctx: context.Context - for cancellation and timeouts")
	fmt.Println("- conversationID: uuid.UUID - the conversation to get messages from")
	fmt.Println("- limit: int - maximum number of messages to return (e.g., 10)")
	fmt.Println("- offset: int - number of messages to skip for pagination (e.g., 0)")

	fmt.Println("\n=== Example completed ===")
	fmt.Println("\nTo test with real data:")
	fmt.Println("1. Start Milvus server: docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest")
	fmt.Println("2. Uncomment the client connection and function call code")
	fmt.Println("3. Run: go run chatbot.go")
}
